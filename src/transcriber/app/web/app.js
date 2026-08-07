'use strict';

/* Интерфейс намеренно без фреймворка: состояние здесь — это выбранный источник
   и текущий список сегментов, ради двух сущностей тянуть библиотеку незачем.
   Всё, что дольше мгновения, считает Python и присылает сюда событиями. */

const $ = (id) => document.getElementById(id);

const state = {
  target: null,     // путь к файлу или ссылка
  segments: [],
  rows: [],         // DOM-строки, в том же порядке что и segments
  active: -1,
  busy: false,
  speakers: null,   // чем размечаются спикеры и предлагать ли pyannote
};

/* --- тема --- */

/* Саму тему выставляет короткий скрипт в <head> — до первой отрисовки, иначе
   при каждом запуске моргала бы бумага перед тёмным. Здесь только переключение. */
function applyTheme(name) {
  document.documentElement.dataset.theme = name;
  $('theme').textContent = name === 'dark' ? '☀' : '☾';
  try { localStorage.setItem('theme', name); } catch { /* приватный режим */ }
}

applyTheme(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');

$('theme').addEventListener('click', () => {
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
});

/* Ссылку открывает система, а не окно: webview здесь — само приложение, и уход
   на страницу означал бы потерю открытого транскрипта.

   Перехват общий, на весь документ: ссылок в интерфейсе больше одной, и забыть
   повесить обработчик на новую — значит однажды увести окно на huggingface.co
   вместе с несохранённым транскриптом. */
document.addEventListener('click', (e) => {
  const link = e.target.closest('a[href^="http"]');
  if (!link) return;
  e.preventDefault();
  window.pywebview.api.open_url(link.href);
});

/* --- служебное --- */

const clock = (seconds) => {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = h ? String(m).padStart(2, '0') : String(m);
  return `${h ? h + ':' : ''}${mm}:${String(s).padStart(2, '0')}`;
};

let toastTimer = null;
function toast(text, bad = false) {
  const node = $('toast');
  node.textContent = text;
  node.classList.toggle('bad', bad);
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 3200);
}

function log(text, bad = false) {
  const lines = $('log-lines');
  const line = document.createElement('div');
  if (bad) line.className = 'err';
  line.textContent = text;
  lines.append(line);
  lines.scrollTop = lines.scrollHeight;
  $('log').hidden = false;
}

/* Чем размечаются спикеры и как это поменять.

   Развёрнуто при первом запуске, дальше — одна строка. Предложение выводится из
   отсутствия токена, поэтому стёртый токен вернёт его обратно, а введённый
   уберёт даже после отказа. */
function renderSpeakers(info) {
  state.speakers = info;
  const box = $('speakers-setup');
  const offer = $('speakers-offer');
  box.hidden = false;
  offer.hidden = !info.offer;

  $('speakers-now').textContent = info.backend === 'pyannote'
    ? 'Speakers: pyannote — brief remarks kept apart'
    : 'Speakers: works out of the box, brief remarks may be merged';
  $('speakers-toggle').textContent = offer.hidden
    ? (info.backend === 'pyannote' ? 'change token' : 'use pyannote')
    : 'hide';
  $('speakers-token').value = '';
}

async function applyToken() {
  const value = $('speakers-token').value.trim();
  if (!value) return;
  renderSpeakers(await window.pywebview.api.save_token(value));
  toast('pyannote will label the speakers from now on');
}

/* Строка о вышедшей версии. Именно строка, а не модальное окно: обновление —
   это предложение, а не событие, ради которого стоит прерывать работу. Заметки
   к релизу уходят в подсказку, чтобы шапка не разрасталась. */
function offerUpdate(version, notes) {
  const box = $('update');
  box.textContent = '';
  box.hidden = false;
  box.title = notes || '';

  const label = document.createElement('span');
  label.textContent = `version ${version} available`;

  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = 'Install';
  button.addEventListener('click', async () => {
    button.disabled = true;
    label.textContent = `installing ${version}…`;
    /* Отказ сразу — предлагать нечего. Отказ по ходу приезжает событием
       `update-failed`, которое перерисует строку целиком. */
    if (!(await window.pywebview.api.install_update())) {
      button.disabled = false;
      label.textContent = `version ${version} available`;
    }
  });

  box.append(label, button);
}

/* Доля распознанного. Полоса появляется только когда работа началась: до этого
   идут подготовка звука, VAD и загрузка модели, а сколько они займут, заранее
   неизвестно — пустая шкала там обещала бы то, чего никто не считал. */
function advance(done) {
  const bar = $('advance');
  bar.hidden = false;
  bar.querySelector('i').style.width = `${Math.min(100, done * 100).toFixed(1)}%`;
}

function setBusy(busy) {
  state.busy = busy;
  /* Пока идёт работа, кнопка не гаснет, а становится «Stop». Гасить её значило
     бы, что часовую запись нельзя передумать — только закрыть окно.

     Остановка не разрушительна: распознанные порции остаются в кэше, и повторный
     запуск продолжит с того же места. Поэтому подтверждения не спрашиваем. */
  $('start').disabled = false;
  $('start').classList.toggle('stopping', busy);
  $('start').textContent = busy ? 'Stop' : 'Transcribe';
  if (!busy) {
    $('advance').hidden = true;
    $('advance').querySelector('i').style.width = '0';
  }
  document.querySelectorAll('.actions button, .weights button').forEach((b) => {
    b.disabled = busy;
  });
}

/* --- выбор источника --- */

function chooseTarget(value, label) {
  state.target = value;
  const node = $('chosen');
  node.textContent = label;
  node.hidden = false;
}

$('pick').addEventListener('click', async () => {
  const path = await window.pywebview.api.pick_file();
  if (path) {
    $('url').value = '';
    chooseTarget(path, path.split('/').pop());
  }
});

$('url').addEventListener('input', (e) => {
  const value = e.target.value.trim();
  if (value) chooseTarget(value, value);
});

const drop = $('drop');
['dragenter', 'dragover'].forEach((type) =>
  drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((type) =>
  drop.addEventListener(type, () => drop.classList.remove('over')));

/* Сам путь к брошенному файлу приходит событием из Python: браузерный объект
   File его не содержит, полное имя подставляет pywebview на своей стороне. */
drop.addEventListener('drop', (e) => e.preventDefault());

$('diarize').addEventListener('change', (e) => {
  $('speakers-field').hidden = !e.target.checked;
});

$('language').addEventListener('change', (e) => {
  $('lang-warn').hidden = !e.target.value;
});

/* --- запуск --- */

$('start').addEventListener('click', async () => {
  /* Та же кнопка останавливает начатое: пока идёт работа, запускать нечего, а
     передумать — единственное, чего может хотеться. */
  if (state.busy) {
    $('start').textContent = 'Stopping…';
    await window.pywebview.api.stop_transcription();
    return;
  }
  if (!state.target) { toast('choose a file or paste a link first', true); return; }
  $('log-lines').replaceChildren();
  setBusy(true);
  const started = await window.pywebview.api.transcribe(state.target, {
    language: $('language').value,
    model: $('model').value,
    diarize: $('diarize').checked,
    speakers: Number($('speakers').value) || null,
  });
  if (!started) setBusy(false);
});

/* --- транскрипт --- */

function renderTranscript(payload) {
  state.segments = payload.segments;
  state.active = -1;

  /* Оговорка стоит рядом с ярлыками, а не в документации: измерено, что sherpa
     сливает короткие реплики с основным голосом — вопрос из зала достанется
     докладчику. Знать об этом надо в тот момент, когда читаешь имена. */
  const note = $('labelled-by');
  note.hidden = payload.labelled_by !== 'sherpa';
  note.textContent = 'Labels from the open models: a brief question from the room '
    + 'may be merged into the speaker who was talking.';

  const box = $('transcript');
  box.replaceChildren();
  state.rows = payload.segments.map((segment) => {
    const row = document.createElement('div');
    row.className = 'row';

    const time = document.createElement('span');
    time.className = 'time';
    time.textContent = clock(segment.start);

    const said = document.createElement('span');
    said.className = 'said';
    if (segment.speaker) {
      const who = document.createElement('span');
      who.className = 'who';
      who.textContent = segment.speaker;
      said.append(who);
    }
    said.append(document.createTextNode(segment.text));

    row.append(time, said);
    row.addEventListener('click', () => {
      $('audio').currentTime = segment.start;
      $('audio').play();
    });
    box.append(row);
    return row;
  });

  const parts = [payload.language, clock(payload.duration)];
  if (payload.speakers?.length) parts.push(`${payload.speakers.length} speakers`);
  $('result-meta').textContent = parts.join(' · ');
  $('result').hidden = false;
}

/* Подсветка текущей реплики. Сегменты упорядочены, поэтому обычно достаточно
   проверить соседей — полный обход нужен только после перемотки. */
function highlight(now) {
  const current = state.active;
  if (current >= 0 && inside(current, now)) return;
  if (current + 1 < state.segments.length && inside(current + 1, now)) {
    return select(current + 1);
  }
  select(state.segments.findIndex((s) => now >= s.start && now < s.end));
}

const inside = (index, now) => {
  const s = state.segments[index];
  return s && now >= s.start && now < s.end;
};

function select(index) {
  if (index === state.active) return;
  state.rows[state.active]?.classList.remove('active');
  state.active = index;
  const row = state.rows[index];
  if (row) {
    row.classList.add('active');
    row.scrollIntoView({ block: 'nearest' });
  }
}

$('audio').addEventListener('timeupdate', (e) => highlight(e.target.currentTime));

/* --- действия над результатом --- */

/* Путь показывается сокращённым: домашний каталог заменяется на «~», иначе
   строка вытесняет всё остальное. */
const shortPath = (path) => path.replace(/^\/Users\/[^/]+/, '~');

function setFolder(path) {
  $('folder').textContent = shortPath(path);
  $('folder').dataset.full = path;
}

$('folder').addEventListener('click', async () => {
  const chosen = await window.pywebview.api.pick_folder();
  if (chosen) {
    setFolder(chosen);
    toast('files will be saved here');
  }
});

document.querySelectorAll('[data-export]').forEach((button) => {
  button.addEventListener('click', async () => {
    try {
      const path = await window.pywebview.api.export(button.dataset.export);
      toast(`saved: ${shortPath(path)}`);
      window.pywebview.api.reveal(path);
    } catch (error) {
      toast(String(error), true);
    }
  });
});

/* Показанное и сохраняемое — одно и то же, поэтому переключатель ходит в Python,
   а не прячет текст на стороне окна. */
function showToggle(visible, which = 'original') {
  $('toggle').hidden = !visible;
  document.querySelectorAll('[data-show]').forEach((b) => {
    b.classList.toggle('on', b.dataset.show === which);
  });
}

document.querySelectorAll('[data-show]').forEach((button) => {
  button.addEventListener('click', () => window.pywebview.api.show(button.dataset.show));
});

document.querySelectorAll('[data-translate]').forEach((button) => {
  button.addEventListener('click', async () => {
    setBusy(true);
    if (!await window.pywebview.api.translate(button.dataset.translate)) setBusy(false);
  });
});

$('summarize').addEventListener('click', async () => {
  setBusy(true);
  if (!await window.pywebview.api.summarize('ru')) setBusy(false);
});

/* --- история --- */

const when = (iso) => {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  const today = new Date();
  const sameDay = date.toDateString() === today.toDateString();
  const time = date.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit' });
  if (sameDay) return `today, ${time}`;
  return `${date.toLocaleDateString('en', { day: 'numeric', month: 'long' })}, ${time}`;
};

function renderHistory(entries) {
  $('history-panel').hidden = entries.length === 0;
  $('history-count').textContent = entries.length ? `${entries.length}` : '';

  $('history').replaceChildren(...entries.map((entry) => {
    const row = document.createElement('article');
    row.className = 'note';

    const topic = document.createElement('p');
    topic.className = 'note-topic';
    topic.textContent = entry.topic || entry.title;

    const meta = document.createElement('p');
    meta.className = 'note-meta';
    const bits = [when(entry.added), clock(entry.duration), entry.language];
    if (entry.speakers) bits.push(`${entry.speakers} speakers`);
    meta.textContent = bits.filter(Boolean).join(' · ');

    const origin = document.createElement('p');
    origin.className = 'note-origin';
    origin.textContent = entry.kind === 'url' ? entry.origin : entry.title;

    const drop = document.createElement('button');
    drop.className = 'note-drop';
    drop.type = 'button';
    drop.textContent = '×';
    drop.title = 'Remove from history';
    drop.addEventListener('click', async (event) => {
      event.stopPropagation();
      renderHistory(await window.pywebview.api.forget_history(entry.key));
    });

    row.append(topic, meta, origin, drop);
    row.addEventListener('click', () => window.pywebview.api.open_history(entry.key));
    return row;
  }));
}

/* --- веса моделей --- */

/* Панель показывается, только когда чего-то не хватает. Скачанное перечислять
   незачем: это не настройка, а разовое препятствие, которое надо убрать. */

const ROLES = { asr: 'Speech recognition', llm: 'Translation and summary' };

const gigabytes = (bytes) => `${(bytes / 1024 ** 3).toFixed(1)} GB`;
/* Мегабайты годятся, пока сеть быстрая; на медленной «0.0 MB/s» не говорит
   ничего, поэтому мелкие скорости показываем в килобайтах. */
const perSecond = (bytes) =>
  bytes >= 1024 ** 2 ? `${(bytes / 1024 ** 2).toFixed(1)} MB/s` : `${Math.round(bytes / 1024)} KB/s`;

/* Скорость считается здесь, а не в Python: это способ показать те же байты,
   а не отдельные данные. Мгновенная скорость скачет вместе с сетью, поэтому
   сглаживаем — иначе цифра дёргается и её невозможно прочитать. */
const SMOOTHING = 0.3;
const rates = new Map();

function rate(repo, done) {
  const now = performance.now() / 1000;
  const previous = rates.get(repo);
  rates.set(repo, { done, at: now, speed: previous?.speed ?? 0 });

  if (!previous || now === previous.at) return previous?.speed ?? 0;
  /* Отрицательный прирост бывает после паузы и повторного старта — счётчик
     тогда начинается заново, а не показывает бессмыслицу. */
  const instant = Math.max(0, (done - previous.done) / (now - previous.at));
  const speed = previous.speed ? previous.speed * (1 - SMOOTHING) + instant * SMOOTHING : instant;
  rates.set(repo, { done, at: now, speed });
  return speed;
}

/* Ниже этого порога оценка превращается в «1456631 min left» — цифру, которая
   не помогает, а пугает. Молчание честнее выдуманного срока. */
const MEANINGFUL_SPEED = 32 * 1024;

/* Точность здесь ни к чему: человеку нужно решить, ждать или заняться другим. */
function timeLeft(bytes, speed) {
  if (speed < MEANINGFUL_SPEED || bytes <= 0) return '';
  const seconds = bytes / speed;
  if (seconds < 90) return `${Math.ceil(seconds)} s left`;
  if (seconds < 5400) return `${Math.ceil(seconds / 60)} min left`;
  return `${(seconds / 3600).toFixed(1)} h left`;
}

function renderWeights(items) {
  const missing = items.filter((item) => !item.ready);
  $('weights').hidden = missing.length === 0;

  $('weights-list').replaceChildren(...missing.map((item) => {
    const row = document.createElement('div');
    row.className = 'weight';
    row.dataset.repo = item.repo;

    const name = document.createElement('span');
    name.className = 'weight-name';
    name.textContent = `${ROLES[item.role] || item.role} · ${item.title}`;

    const note = document.createElement('span');
    note.className = 'weight-note';
    /* Ненулевой размер у нескачанной модели означает оборванную попытку. С места
       обрыва она не продолжится: загрузчик заводит временный файл заново. */
    note.textContent = item.size ? `stopped at ${gigabytes(item.size)}` : 'not downloaded';

    /* Одна кнопка на два состояния: пока качаем — «пауза», иначе — «скачать».
       Две кнопки рядом заставляли бы выбирать там, где выбора нет. */
    const button = document.createElement('button');
    button.className = 'ghost';
    button.dataset.mode = 'download';
    button.textContent = item.size ? 'Download again' : 'Download';
    button.addEventListener('click', async () => {
      if (button.dataset.mode === 'pause') {
        await window.pywebview.api.pause_download();
        return;
      }
      if (!await window.pywebview.api.download_weights(item.repo)) {
        toast('wait for the current job to finish', true);
      }
    });

    const bar = document.createElement('div');
    bar.className = 'weight-bar';
    bar.hidden = true;
    bar.append(document.createElement('i'));

    const head = document.createElement('div');
    head.className = 'weight-head';
    head.append(name, note, button);

    row.append(head, bar);
    return row;
  }));
}

function weightRow(repo) {
  return $('weights-list').querySelector(`[data-repo="${CSS.escape(repo)}"]`);
}

function weightButton(repo, mode, label) {
  const button = weightRow(repo)?.querySelector('button');
  if (!button) return;
  button.dataset.mode = mode;
  button.textContent = label;
}

function weightProgress(repo, done, total) {
  const row = weightRow(repo);
  if (!row) return;

  const bar = row.querySelector('.weight-bar');
  bar.hidden = false;

  const speed = rate(repo, done);
  const parts = [];
  /* Пока размер модели неизвестен (сеть промолчала на запрос), долю показать
     нечестно — но и застывшая пустая полоса врёт, будто ничего не происходит.
     Поэтому она в этом случае просто движется. */
  bar.classList.toggle('unknown', !total);
  if (total) {
    bar.querySelector('i').style.width = `${Math.min(100, (done / total) * 100).toFixed(1)}%`;
    parts.push(`${gigabytes(done)} of ${gigabytes(total)}`);
  } else {
    parts.push(`${gigabytes(done)} downloaded`);
  }
  if (speed) parts.push(perSecond(speed));
  if (total) {
    const left = timeLeft(total - done, speed);
    if (left) parts.push(left);
  }
  row.querySelector('.weight-note').textContent = parts.join(' · ');
}

/* --- события из Python --- */

window.appEvent = (event) => {
  switch (event.kind) {
    case 'progress':
      log(event.message);
      break;
    case 'advance':
      advance(event.done);
      break;
    case 'update-found':
    /* Неудача перерисовывает строку заново — с рабочей кнопкой. Иначе она
       осталась бы висеть в «installing…» до перезапуска. */
    case 'update-failed': // eslint-disable-line no-fallthrough
      offerUpdate(event.version, event.notes);
      break;
    case 'update-installed':
      $('update').textContent = `version ${event.version} installed — restart to finish`;
      toast('restart the application to finish the update');
      break;
    case 'transcript':
      $('audio').src = event.audio;
      $('audio').hidden = !event.audio;
      renderTranscript(event);
      showToggle(false);
      $('summary').hidden = true;
      break;
    case 'stopped':
      log('stopped — what was recognised is kept, running again resumes from there');
      toast('stopped; the finished parts are kept');
      setBusy(false);
      break;
    case 'history':
      renderHistory(event.entries);
      break;
    case 'dropped':
      $('url').value = '';
      chooseTarget(event.path, event.name);
      break;
    case 'translation':
      renderTranscript(event);
      showToggle(true, 'translation');
      toast('translation ready — this is what gets saved');
      break;
    case 'shown':
      renderTranscript(event);
      showToggle(true, event.which);
      break;
    case 'summary':
      $('summary').textContent = event.markdown;
      $('summary').hidden = false;
      break;
    case 'weights-needed':
      renderWeights(event.items);
      break;
    case 'weights-started':
      log('downloading weights — this happens once');
      weightButton(event.repo, 'pause', 'Pause');
      break;
    case 'weights-progress':
      weightProgress(event.repo, event.done, event.total);
      break;
    case 'weights-paused':
      /* Начатый кусок остаётся на диске, но послужить продолжением не сможет:
         следующая попытка качает файл заново. Поэтому не «продолжить». */
      weightButton(event.repo, 'download', 'Download again');
      toast('stopped — the next attempt starts over');
      break;
    case 'weights-ready':
      renderWeights(event.items);
      toast('model ready — you can transcribe now');
      break;
    case 'error':
      log(event.message, true);
      toast(event.message, true);
      break;
    case 'idle':
      setBusy(false);
      break;
  }
};

window.addEventListener('pywebviewready', async () => {
  const setup = await window.pywebview.api.setup();

  /* В шапке — устройство, а не имя модели: модель выбирается тут же в форме,
     а вот на чём идёт счёт, иначе неоткуда узнать. Это же первое, что нужно,
     когда распознавание вдруг стало медленным. */
  $('engine').textContent = setup.device;
  $('engine').title = `Translation and summary: ${setup.llm}`;
  setFolder(setup.output);
  renderSpeakers(setup.speakers);

  $('speakers-toggle').addEventListener('click', () => {
    const offer = $('speakers-offer');
    offer.hidden = !offer.hidden;
    $('speakers-toggle').textContent = offer.hidden
      ? (state.speakers.backend === 'pyannote' ? 'change token' : 'use pyannote')
      : 'hide';
  });
  $('speakers-save').addEventListener('click', applyToken);
  $('speakers-token').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') applyToken();
  });
  $('speakers-skip').addEventListener('click', async () => {
    renderSpeakers(await window.pywebview.api.decline_speakers());
  });

  renderWeights(await window.pywebview.api.weights_status());
  renderHistory(await window.pywebview.api.history());

  const select = $('model');
  select.replaceChildren(...setup.models.map(({ name, hint }) => {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = hint ? `${name} — ${hint}` : name;
    option.selected = name === setup.current;
    return option;
  }));

  /* Последним и без ожидания: единственный запрос наружу, и окно не должно
     зависеть от того, ответил ли GitHub. Если новее ничего нет — не будет и
     события, шапка останется прежней. */
  window.pywebview.api.check_update();
});

// aura frontend — shared state, DOM references, and injected config.
//
// Loaded FIRST (before any other aura *.js) so every later script can
// see the `var` declarations here. We intentionally use `var` (not
// let/const) because classic <script src> files do NOT share block
// scope; only `var` and `function` attach to the global so that
// later scripts can read them.

var VIDEO_SEND_INTERVAL = (window.APP_CONFIG && window.APP_CONFIG.videoSendInterval) || 1.0;
var VIDEO_FPS = 15;
var videoStream = null;
var audioRecorder = null;
var audioStream = null;
var audioChunks = [];
var isVideoStreaming = false;
var isAudioRecording = false;
var frameCount = 0;
var totalDataSize = 0;
var audioCount = 0;
var audioStartTime = null;
var pollInterval = null;
var currentStreamingBubble = null;
var currentStreamingResponseId = null;
var streamedResponseIds = new Set();
var currentFacingMode = 'environment';
var currentSessionId = null;
var currentDisplayedResponseId = null;
var canceledResponseIds = new Set();
var videoMediaRecorder = null;
var videoRecordTimeout = null;

// Canvas 降采样 (原生分辨率显示, 640x480 发送)
var SEND_WIDTH = 640;
var SEND_HEIGHT = 480;
var sendCanvas = null;
var sendCanvasCtx = null;
var sendStream = null;
var eventSource = null;
var canvasDrawInterval = null;

// Web Audio API for streaming PCM playback.
var webAudioContext = null;
var webAudioChunkQueue = [];
var webAudioIsPlaying = false;
var webAudioCurrentResponseId = null;
var webAudioCurrentSentenceIdx = -1;
var webAudioSampleRate = 24000;
var webAudioNextPlayTime = 0;
var webAudioActiveSources = [];

// DOM elements.
var videoPreview = document.getElementById('videoPreview');
var startVideoBtn = document.getElementById('startVideoBtn');
var stopVideoBtn = document.getElementById('stopVideoBtn');
var micBtn = document.getElementById('micBtn');
var micHint = document.getElementById('micHint');
var waveform = document.getElementById('waveform');
var liveStatus = document.getElementById('liveStatus');
var sendingStatus = document.getElementById('sendingStatus');
var responsePanel = document.getElementById('responsePanel');
var responseContent = document.getElementById('responseContent');

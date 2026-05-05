// aura frontend — page-level init (onload hooks + audio context unlock).

// 自动播放解锁状态
var audioContextUnlocked = false;

// 解锁音频播放（解决浏览器自动播放限制）
// Web Audio API 也有自己的 unlockWebAudioContext，两者独立
function unlockAudioContext() {
    if (audioContextUnlocked) return;
    audioContextUnlocked = true;
}


// 页面加载完成
window.onload = function() {
    log('系统初始化完成 (Streaming 版本)', 'success');
    log('点击 Start 开始视频流传输', 'info');
    log('按住 Record 进行语音录制', 'info');
    log('📝 流式 Token 输出已启用', 'info');
    log('🔊 Web Audio API 流式 PCM 播放已启用', 'info');

    // 初始化录音按钮事件（避免内联事件导致的问题）
    initMicButton();

    // 添加页面点击事件来解锁音频（作为备选方案）
    document.addEventListener('click', unlockAudioContext, { once: true });
    document.addEventListener('touchstart', unlockAudioContext, { once: true });
};


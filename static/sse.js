// aura frontend — SSE event stream and streaming-token handler.

// =============== SSE Event Stream ===============

function startEventStream() {
    stopEventStream();
    const url = `/api/events?session_id=${encodeURIComponent(currentSessionId || '')}`;
    eventSource = new EventSource(url);

    eventSource.addEventListener('token', (e) => {
        try {
            const payload = JSON.parse(e.data);
            // payload.raw is the JSON token envelope produced by the inference server
            const tokenData = JSON.parse(payload.raw);
            handleStreamingToken(tokenData);
        } catch (err) {
            console.error('Failed to parse token event:', err, e.data);
        }
    });

    eventSource.addEventListener('chunk', (e) => {
        try {
            const c = JSON.parse(e.data);
            playPcmChunk(
                c.pcm_base64,
                c.sample_rate,
                c.response_id,
                c.sentence_idx,
                c.chunk_idx,
                c.is_final
            );
        } catch (err) {
            console.error('Failed to parse chunk event:', err);
        }
    });

    eventSource.addEventListener('error', (e) => {
        // Named "error" events carry server-reported errors (session rejection etc).
        // EventSource also emits onerror() for network failures — those have no data.
        if (e.data) {
            try {
                const { message } = JSON.parse(e.data);
                log(`⚠️ 服务端错误: ${message}`, 'error');
                alert(`连接被拒绝: ${message}`);
                stopVideo();
            } catch (err) {
                console.error('Failed to parse error event:', err);
            }
        }
    });

    eventSource.addEventListener('close', () => {
        stopEventStream();
    });

    eventSource.onerror = (err) => {
        // Network-level failure. EventSource will auto-reconnect by default.
        console.warn('EventSource transport error (will retry):', err);
    };
}

function stopEventStream() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    // Stop any Web Audio playback that was being fed by chunks.
    stopAllWebAudioPlayback(true);
    webAudioCurrentResponseId = null;
}

// 处理单个流式 token
function handleStreamingToken(tokenData) {
    const { response_id, token, is_final, type, query, is_start, is_silent } = tokenData;

    const floatingContainer = document.getElementById('floatingResponses');
    if (!floatingContainer) return;

    // 辅助函数：创建聊天气泡
    function createChatBubble(text, bubbleType, icon) {
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${bubbleType}`;
        bubble.innerHTML = `<span class="bubble-icon">${icon}</span>${text}`;
        return bubble;
    }

    // Plan 2: ASR query echo (Type 10) - 立即显示用户文字，不创建助手气泡
    if (type === 'asr_query') {
        const q = tokenData.query;
        if (q && q.trim()) {
            const formattedQuery = q.replace(/\n/g, '<br>');
            const userBubble = createChatBubble(formattedQuery, 'user', '🗣️');
            floatingContainer.appendChild(userBubble);
            floatingContainer.scrollTop = floatingContainer.scrollHeight;
            log(`🗣️ 用户: ${q.substring(0, 30)}...`, 'info');
        }
        return;
    }

    if (type !== 'streaming_token') return;

    // 如果是 silent 响应，不显示任何内容，只重置状态
    if (is_silent) {
        // 如果有正在显示的流式气泡，移除它
        if (currentStreamingBubble && currentStreamingBubble.parentNode) {
            currentStreamingBubble.remove();
        }
        currentStreamingBubble = null;
        currentStreamingResponseId = null;
        return;
    }

    // 检查是否是新的响应
    if (currentStreamingResponseId !== response_id) {
        // 新的响应开始，创建新的流式气泡
        currentStreamingResponseId = response_id;
        currentDisplayedResponseId = response_id; // 更新当前显示的响应 ID，用于 TTS 匹配

        // 如果有 query (ASR 结果)，先显示用户的消息（兼容旧协议）
        if (is_start && query && query.trim()) {
            const formattedQuery = query.replace(/\n/g, '<br>');
            const userBubble = createChatBubble(formattedQuery, 'user', '🗣️');
            floatingContainer.appendChild(userBubble);
            log(`🗣️ 用户: ${query.substring(0, 30)}...`, 'info');
        }

        // 创建流式显示气泡
        currentStreamingBubble = document.createElement('div');
        currentStreamingBubble.className = 'chat-bubble assistant streaming';
        currentStreamingBubble.innerHTML = `<span class="bubble-icon">🤖</span><span class="streaming-content"></span><span class="streaming-cursor">▊</span>`;
        floatingContainer.appendChild(currentStreamingBubble);

        // 滚动到底部
        floatingContainer.scrollTop = floatingContainer.scrollHeight;

        log(`📝 开始流式输出 (id: ${response_id.substring(0, 8)}...)`, 'info');
    }

    // 追加 token 到当前气泡
    if (currentStreamingBubble) {
        const contentEl = currentStreamingBubble.querySelector('.streaming-content');
        const cursorEl = currentStreamingBubble.querySelector('.streaming-cursor');

        if (contentEl && token) {
            // 过滤掉 <|silent|> 等特殊标记
            if (token.includes('<|silent|>') || token.includes('<|silent|')) {
                return;  // 跳过 silent 标记
            }
            // 替换换行符并追加文本，使用 insertAdjacentHTML 比 innerHTML += 更高效
            const formattedToken = token.replace(/\n/g, '<br>');
            contentEl.insertAdjacentHTML('beforeend', formattedToken);
        }

        // 如果是最后一个 token，移除光标
        if (is_final) {
            if (cursorEl) cursorEl.remove();
            currentStreamingBubble.classList.remove('streaming');

            log(`✅ 流式输出完成`, 'success');

            // 记录已流式显示的响应 ID（防止重复）
            streamedResponseIds.add(response_id);
            // 10秒后清理，防止集合无限增长
            setTimeout(() => streamedResponseIds.delete(response_id), 10000);

            // 重置流式状态
            currentStreamingBubble = null;
            currentStreamingResponseId = null;
        }
    }
}

// =============== 流式 Token 功能结束 ===============

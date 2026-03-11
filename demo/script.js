document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const micBtn = document.getElementById('micBtn');
    const fileInput = document.getElementById('fileInput');
    const fileStatus = document.getElementById('fileStatus');
    const fileName = document.getElementById('fileName');
    const removeFileBtn = document.getElementById('removeFileBtn');
    const submitBtn = document.getElementById('submitBtn');
    const uploadArea = document.getElementById('uploadArea');
    
    // Panels
    const inputPanel = document.getElementById('inputPanel');
    const loadingPanel = document.getElementById('loadingPanel');
    const resultPanel = document.getElementById('resultPanel');
    
    // Loading components
    const loadingText = document.getElementById('loadingText');
    const progressBar = document.getElementById('progressBar');
    const resetBtn = document.getElementById('resetBtn');

    // Visualization
    const vizLayout = document.getElementById('vizLayout');
    const vecLineB = document.getElementById('vecLineB');
    const angleText = document.getElementById('angleText');
    const cosText = document.getElementById('cosText');
    const consistencyText = document.getElementById('consistencyText');
    const alignmentMetrics = document.getElementById('alignmentMetrics');
    const diffHeatmap = document.getElementById('diffHeatmap');
    const heatmapInspector = document.getElementById('heatmapInspector');
    const topkList = document.getElementById('topkList');
    // State
    let isRecording = false;
    let audioFile = null;
    let recordingTimeout = null;

    // ----- File Upload & Drag-Drop -----
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (isRecording) stopRecording();

        if (file.type.startsWith('audio/') || file.type.startsWith('video/')) {
            audioFile = file;
            fileName.textContent = file.name;
            fileStatus.classList.remove('hidden');
            submitBtn.disabled = false;
        } else {
            alert('请上传有效的音频或视频文件。');
        }
    }

    removeFileBtn.addEventListener('click', () => {
        audioFile = null;
        fileInput.value = '';
        fileStatus.classList.add('hidden');
        submitBtn.disabled = true;
    });

    // ----- Microphone Recording (Simulation/Real) -----
    
    micBtn.addEventListener('click', async () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    function startRecording() {
        isRecording = true;
        micBtn.classList.add('recording');
        micBtn.innerHTML = '<i class="fas fa-stop"></i>';
        
        if (audioFile) {
            removeFileBtn.click();
        }

        submitBtn.disabled = true;
        submitBtn.innerHTML = '正在录音... <i class="fas fa-microphone-lines"></i>';

        recordingTimeout = setTimeout(() => {
            if (isRecording) stopRecording();
        }, 5000); // 录制5秒后自动停止
    }

    function stopRecording() {
        isRecording = false;
        clearTimeout(recordingTimeout);
        
        micBtn.classList.remove('recording');
        micBtn.innerHTML = '<i class="fas fa-microphone"></i>';
        
        fileName.textContent = `语料录制_${new Date().getTime().toString().slice(-6)}.wav`;
        fileStatus.classList.remove('hidden');
        audioFile = { name: fileName.textContent }; // Mock file
        
        submitBtn.disabled = false;
        submitBtn.innerHTML = '开始深度分析 <i class="fas fa-arrow-right"></i>';
    }

    // ----- Analysis Process -----
    
    submitBtn.addEventListener('click', async () => {
        if (!audioFile) return;

        inputPanel.classList.remove('active');
        setTimeout(() => {
            inputPanel.style.display = 'none';
            loadingPanel.style.display = 'block';
            
            requestAnimationFrame(() => {
                loadingPanel.classList.add('active');
            });
            
            // Starts real analysis via backend API
            performRealAnalysis(audioFile);
        }, 400); 
    });

    async function performRealAnalysis(file) {
        // Mock progress updates while waiting
        loadingText.textContent = "上传音频至后台...";
        progressBar.style.width = '10%';
        
        try {
            const formData = new FormData();
            formData.append('file', file);

            // Trigger AI loading visually
            let loadInterval = setInterval(() => {
                let currentProg = parseInt(progressBar.style.width) || 10;
                if(currentProg < 85) progressBar.style.width = (currentProg + 2) + '%';
                if(currentProg > 20) loadingText.textContent = "提取声学与韵律特征 (Extracting Features)...";
                if(currentProg > 50) loadingText.textContent = "大语言模型语义投影 (Semantic Mapping)...";
                if(currentProg > 80) loadingText.textContent = "多模态偏移计算 (Deviation Analysis)...";
            }, 300);

            // Fetch from FastAPI
            const response = await fetch('http://127.0.0.1:8000/analyze', {
                method: 'POST',
                body: formData
            });

            clearInterval(loadInterval);
            progressBar.style.width = '100%';
            loadingText.textContent = "分析完成！";

            const data = await response.json();
            
            if (data.status === 'success' || data.status === 'mock_success') {
                setTimeout(() => showResult(data), 500);
            } else {
                alert("后端分析错误：" + (data.message || "未知错误"));
                resetToInput();
            }
        } catch (error) {
            console.error("API Fetch Error:", error);
            alert("请求后端失败，请确保后台 FastAPI 服务正在运行 (cd backend && uvicorn app:app --reload)");
            resetToInput();
        }
    }

    // ----- Result Presentation -----
    
    function showResult(data) {
        loadingPanel.classList.remove('active');
        
        // 更新 DOM 结果
        const statusBadge = document.getElementById('resultStatusBadge');
        const insightText = document.getElementById('resultInsight');
        
        if (data.is_high_risk) {
            statusBadge.className = "status-badge critical";
            statusBadge.innerHTML = '<i class="fas fa-exclamation-triangle"></i> 高风险发现 (分数: ' + data.deviation_score + ')';
            insightText.innerHTML = `
                系统通过 <strong>Speech2Vec</strong> 深度模型分析检测到目标的 <em>语义表达 (Dim: ${data.features.semantic_dim})</em> 与 <em>韵律特征 (Dim: ${data.features.prosody_dim})</em> 之间存在 <span class="highlight">显著偏移 (Deviation: ${data.deviation_score})</span>。<br>
                这通常表现在传递表面意思时伴随了反常的声学高频抖动（如非自然的音调/语速突变）。<br>
                <em>结论支持：极高概率存在隐瞒真实情感或认知失调的状况。</em>`;
        } else {
            statusBadge.className = "status-badge safe";
            statusBadge.style.background = "var(--success-color)";
            statusBadge.style.color = "white";
            statusBadge.innerHTML = '<i class="fas fa-check-circle"></i> 表现自然 (分数: ' + data.deviation_score + ')';
            insightText.innerHTML = `
                系统综合 <em>语义编码器 (${data.features.semantic_dim}维)</em> 和 <em>韵律编码器 (${data.features.prosody_dim}维)</em> 得出结论，两者的多层激活方差偏移度较低 (<span style="color:var(--success-color); font-weight:bold;">Deviation: ${data.deviation_score}</span>)。<br>
                目标的言语内容与潜意识声学特征保持高度一致。<br>
                <em>结论支持：当前样本表现自然，未检测到掩饰情绪的明显迹象。</em>`;
        }

        renderVisualizations(data);

        setTimeout(() => {
            loadingPanel.style.display = 'none';
            resultPanel.style.display = 'block';
            
            progressBar.style.width = '0%'; 
            
            requestAnimationFrame(() => {
                resultPanel.classList.add('active');
            });
        }, 400);
    }
    // ----- Visualization -----

    function clampNumber(value, min, max) {
        if (typeof value !== 'number' || Number.isNaN(value)) return min;
        return Math.max(min, Math.min(max, value));
    }

    function formatNumber(value, digits = 3) {
        if (typeof value !== 'number' || Number.isNaN(value)) return '--';
        return value.toFixed(digits);
    }

    function computeCosine(a, b, n) {
        let dot = 0;
        let na = 0;
        let nb = 0;
        for (let i = 0; i < n; i++) {
            const av = Number(a[i]);
            const bv = Number(b[i]);
            if (!Number.isFinite(av) || !Number.isFinite(bv)) continue;
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        const denom = Math.sqrt(na) * Math.sqrt(nb) + 1e-12;
        return dot / denom;
    }

    function clearVisualizations() {
        if (alignmentMetrics) alignmentMetrics.innerHTML = '';
        if (diffHeatmap) diffHeatmap.innerHTML = '';
        if (topkList) topkList.innerHTML = '';
        if (heatmapInspector) heatmapInspector.textContent = '将鼠标悬停在格子上查看数值';
        if (angleText) angleText.textContent = 'Angle: --';
        if (cosText) cosText.textContent = 'cos: --';
        if (consistencyText) consistencyText.textContent = 'consistency: --';
        if (vecLineB) vecLineB.setAttribute('transform', 'rotate(90 80 70)');
    }

    function renderAngleGauge({ deviation, cosineSimilarity, angleDegrees, consistency, dims }) {
        if (vecLineB) vecLineB.setAttribute('transform', `rotate(${angleDegrees} 80 70)`);
        if (angleText) angleText.textContent = `Angle: ${formatNumber(angleDegrees, 1)}°`;
        if (cosText) cosText.textContent = `cos: ${formatNumber(cosineSimilarity, 3)}`;
        if (consistencyText) consistencyText.textContent = `consistency: ${formatNumber(consistency, 3)}`;

        if (!alignmentMetrics) return;
        const chips = [
            { label: 'deviation', value: formatNumber(deviation, 3), danger: deviation > 0.5 },
            { label: 'angle', value: `${formatNumber(angleDegrees, 1)}deg` },
            { label: 'cos', value: formatNumber(cosineSimilarity, 3) },
            { label: 'dims', value: String(dims) },
        ];

        alignmentMetrics.innerHTML = chips
            .map((c) => `<span class="metric-chip${c.danger ? ' danger' : ''}">${c.label}: ${c.value}</span>`)
            .join('');
    }

    function renderDiffHeatmap(pVec, sVec, n) {
        if (!diffHeatmap) return;
        diffHeatmap.innerHTML = '';

        const diffs = new Array(n);
        for (let i = 0; i < n; i++) {
            const pv = Number(pVec[i]);
            const sv = Number(sVec[i]);
            diffs[i] = Number.isFinite(pv) && Number.isFinite(sv) ? Math.abs(pv - sv) : 0;
        }

        const sorted = diffs.slice().sort((a, b) => a - b);
        const p95 = sorted[Math.floor(sorted.length * 0.95)] ?? sorted[sorted.length - 1] ?? 1;
        const scale = p95 > 0 ? p95 : (sorted[sorted.length - 1] || 1);

        const top = diffs
            .map((d, idx) => ({ idx, d }))
            .sort((a, b) => b.d - a.d)
            .slice(0, 10);

        for (let i = 0; i < n; i++) {
            const intensity = Math.min(1, diffs[i] / (scale + 1e-12));
            const hue = 210 - 210 * intensity; // blue -> red
            const alpha = 0.12 + 0.88 * intensity;

            const cell = document.createElement('div');
            cell.className = 'heat-cell';
            cell.style.backgroundColor = `hsla(${hue}, 90%, 55%, ${alpha})`;
            cell.title = `#${i + 1} | p=${formatNumber(Number(pVec[i]), 3)} s=${formatNumber(Number(sVec[i]), 3)} | diff=${formatNumber(diffs[i], 3)}`;
            cell.addEventListener('mouseenter', () => {
                if (!heatmapInspector) return;
                heatmapInspector.textContent = `dim #${i + 1}: p=${formatNumber(Number(pVec[i]), 3)}  s=${formatNumber(Number(sVec[i]), 3)}  |diff|=${formatNumber(diffs[i], 3)}`;
            });
            diffHeatmap.appendChild(cell);
        }

        if (topkList) {
            topkList.innerHTML = top
                .map((t) => `<span class="chip"><strong>#${t.idx + 1}</strong> diff=${formatNumber(t.d, 3)}</span>`)
                .join('');
        }
    }

    function renderVisualizations(data) {
        if (!vizLayout) return;

        const features = data && data.features ? data.features : null;
        const pVec = features && Array.isArray(features.prosody_vector) ? features.prosody_vector : null;
        const sVec = features && Array.isArray(features.semantic_vector) ? features.semantic_vector : null;

        if (!pVec || !sVec || pVec.length === 0 || sVec.length === 0) {
            clearVisualizations();
            vizLayout.style.display = 'none';
            return;
        }

        vizLayout.style.display = '';

        const n = Math.min(pVec.length, sVec.length);
        let cos = data && data.alignment && typeof data.alignment.cosine_similarity === 'number'
            ? data.alignment.cosine_similarity
            : computeCosine(pVec, sVec, n);

        cos = clampNumber(cos, -1, 1);

        let angle = data && data.alignment && typeof data.alignment.angle_degrees === 'number'
            ? data.alignment.angle_degrees
            : (Math.acos(cos) * 180) / Math.PI;

        angle = clampNumber(angle, 0, 180);

        let consistency = data && data.alignment && typeof data.alignment.consistency === 'number'
            ? data.alignment.consistency
            : (1.0 / (1.0 + Math.exp(-(cos * 40.0))));

        consistency = clampNumber(consistency, 0, 1);

        const deviation = typeof data.deviation_score === 'number' ? data.deviation_score : Number(data.deviation_score);

        renderAngleGauge({
            deviation: Number.isFinite(deviation) ? deviation : 0,
            cosineSimilarity: cos,
            angleDegrees: angle,
            consistency,
            dims: n,
        });

        renderDiffHeatmap(pVec, sVec, n);
    }

    function resetToInput() {
        loadingPanel.classList.remove('active');
        resultPanel.classList.remove('active');
        
        setTimeout(() => {
            loadingPanel.style.display = 'none';
            resultPanel.style.display = 'none';
            inputPanel.style.display = 'block';
            
            progressBar.style.width = '0%';
            removeFileBtn.click();
            submitBtn.innerHTML = '开始深度分析 <i class="fas fa-arrow-right"></i>';
            
            requestAnimationFrame(() => {
                inputPanel.classList.add('active');
            });
        }, 400);
    }

    resetBtn.addEventListener('click', resetToInput);
});

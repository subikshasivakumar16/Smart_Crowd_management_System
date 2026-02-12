/**
 * Environment Monitoring Dashboard - JavaScript
 * Chart.js wave-style chart, sensor cards, risk level, localStorage history
 */

(function() {
    'use strict';

    const STORAGE_KEY = 'sensorHistory';
    const MAX_POINTS = 200;

    let chart = null;
    let historyData = {
        labels: [],
        temp: [],
        hum: [],
        oxy: [],
        sound: []
    };

    // Restore from localStorage
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            const parsed = JSON.parse(stored);
            historyData = {
                labels: parsed.labels || [],
                temp: parsed.temp || [],
                hum: parsed.hum || [],
                oxy: parsed.oxy || [],
                sound: parsed.sound || []
            };
            if (historyData.labels.length > MAX_POINTS) {
                const cut = historyData.labels.length - MAX_POINTS;
                historyData.labels = historyData.labels.slice(cut);
                historyData.temp = historyData.temp.slice(cut);
                historyData.hum = historyData.hum.slice(cut);
                historyData.oxy = historyData.oxy.slice(cut);
                historyData.sound = historyData.sound.slice(cut);
            }
        }
    } catch (e) {}

    function saveHistory() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(historyData));
        } catch (e) {}
    }

    function getChartWidth() {
        var pts = Math.max(50, historyData.labels.length);
        return Math.max(1200, pts * 8);
    }

    function initChart() {
        const ctx = document.getElementById('sensorChart');
        if (!ctx) return;

        const w = getChartWidth();
        ctx.style.width = w + 'px';
        ctx.style.height = '320px';
        ctx.width = w;
        ctx.height = 320;

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: historyData.labels,
                datasets: [
                    {
                        label: 'Temperature °C',
                        data: historyData.temp,
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.08)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    },
                    {
                        label: 'Humidity %',
                        data: historyData.hum,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.08)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    },
                    {
                        label: 'Oxygen %',
                        data: historyData.oxy,
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.08)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    },
                    {
                        label: 'Sound dB',
                        data: historyData.sound,
                        borderColor: '#a855f7',
                        backgroundColor: 'rgba(168, 85, 247, 0.08)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }
                ]
            },
            options: {
                responsive: false,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { position: 'top' },
                    tooltip: { enabled: true }
                },
                scales: {
                    x: {
                        display: true,
                        ticks: { maxTicksLimit: 20 }
                    },
                    y: {
                        display: true,
                        beginAtZero: false,
                        suggestedMin: 0,
                        suggestedMax: 100
                    }
                },
                layout: {
                    padding: { top: 10, right: 10, bottom: 10, left: 10 }
                }
            }
        });
    }

    function updateChart() {
        if (!chart) return;
        chart.data.labels = historyData.labels;
        chart.data.datasets[0].data = historyData.temp;
        chart.data.datasets[1].data = historyData.hum;
        chart.data.datasets[2].data = historyData.oxy;
        chart.data.datasets[3].data = historyData.sound;

        const w = getChartWidth();
        const ctx = chart.canvas;
        if (ctx.width !== w) {
            ctx.style.width = w + 'px';
            ctx.width = w;
        }
        chart.update('none');
    }

    function updateRisk(level, score) {
        const el = document.getElementById('risk-display');
        if (!el) return;
        el.textContent = level;
        el.className = 'risk-badge';
        if (level === 'Normal') el.classList.add('risk-normal');
        else if (level === 'Medium') el.classList.add('risk-medium');
        else el.classList.add('risk-danger');
    }

    function updateZoneSummary(zones) {
        const container = document.getElementById('zone-summary');
        if (!container) return;
        let html = '';
        for (const z of ['A', 'B', 'C', 'D']) {
            const d = zones[z] || {};
            html += `<div class="col-6 col-md-3">
                <div class="border rounded p-2">
                    <strong>Zone ${z}</strong><br>
                    <small>Temp: ${(d.temperature || '--')}°C | Hum: ${(d.humidity || '--')}% | O₂: ${(d.oxygen || '--')}% | Sound: ${(d.sound || '--')}dB</small>
                </div>
            </div>`;
        }
        container.innerHTML = html;
    }

    function fetchData() {
        fetch('/get_data')
            .then(r => r.json())
            .then(data => {
                const zones = data.zones || {};
                const avg = (zones.A && zones.B && zones.C && zones.D) ? {
                    temperature: (zones.A.temperature + zones.B.temperature + zones.C.temperature + zones.D.temperature) / 4,
                    humidity: (zones.A.humidity + zones.B.humidity + zones.C.humidity + zones.D.humidity) / 4,
                    oxygen: (zones.A.oxygen + zones.B.oxygen + zones.C.oxygen + zones.D.oxygen) / 4,
                    sound: (zones.A.sound + zones.B.sound + zones.C.sound + zones.D.sound) / 4
                } : { temperature: 25, humidity: 65, oxygen: 21, sound: 45 };

                document.getElementById('temp-val').textContent = avg.temperature.toFixed(1);
                document.getElementById('hum-val').textContent = avg.humidity.toFixed(1);
                document.getElementById('oxy-val').textContent = avg.oxygen.toFixed(2);
                document.getElementById('sound-val').textContent = avg.sound.toFixed(1);

                updateRisk(data.risk_level || 'Normal', data.risk_score || 0);
                updateZoneSummary(zones);
            })
            .catch(() => {});
    }

    function fetchHistory() {
        fetch('/get_history')
            .then(r => r.json())
            .then(hist => {
                if (!Array.isArray(hist) || hist.length === 0) return;

                const seen = new Set();
                const batch = [];
                for (let i = hist.length - 1; i >= 0 && batch.length < 4; i--) {
                    const ts = hist[i].timestamp;
                    if (!seen.has(ts)) {
                        seen.add(ts);
                        batch.push(hist[i]);
                    }
                }
                if (batch.length === 0) return;

                const t = batch.reduce((a, b) => a + (b.temperature || 25), 0) / batch.length;
                const h = batch.reduce((a, b) => a + (b.humidity || 65), 0) / batch.length;
                const o = batch.reduce((a, b) => a + (b.oxygen || 21), 0) / batch.length;
                const s = batch.reduce((a, b) => a + (b.sound || 45), 0) / batch.length;
                const ts = batch[0].timestamp || new Date().toLocaleTimeString();
                const label = ts.length > 8 ? ts.substring(11, 19) : ts;

                if (historyData.labels.length > 0 && historyData.labels[historyData.labels.length - 1] === label) return;

                historyData.labels.push(label);
                historyData.temp.push(Number(t.toFixed(2)));
                historyData.hum.push(Number(h.toFixed(2)));
                historyData.oxy.push(Number(o.toFixed(2)));
                historyData.sound.push(Number(s.toFixed(2)));

                if (historyData.labels.length > MAX_POINTS) {
                    historyData.labels.shift();
                    historyData.temp.shift();
                    historyData.hum.shift();
                    historyData.oxy.shift();
                    historyData.sound.shift();
                }

                updateChart();
                saveHistory();
            })
            .catch(() => {});
    }

    function initScrollDrag() {
        var el = document.querySelector('.chart-scroll');
        if (!el) return;
        var isDown = false, startX, scrollLeft;
        el.addEventListener('mousedown', function(e) {
            isDown = true;
            el.style.cursor = 'grabbing';
            startX = e.pageX - el.offsetLeft;
            scrollLeft = el.scrollLeft;
        });
        el.addEventListener('mouseleave', function() {
            isDown = false;
            el.style.cursor = 'grab';
        });
        el.addEventListener('mouseup', function() {
            isDown = false;
            el.style.cursor = 'grab';
        });
        el.addEventListener('mousemove', function(e) {
            if (!isDown) return;
            e.preventDefault();
            var x = e.pageX - el.offsetLeft;
            el.scrollLeft = scrollLeft - (x - startX);
        });
        el.style.cursor = 'grab';
    }

    document.getElementById('btn-download-report').addEventListener('click', function(e) {
        e.preventDefault();
        var d = document.getElementById('report-date').value || '';
        window.location.href = '/download/report' + (d ? '?date=' + d : '');
    });
    document.getElementById('btn-download-chart').addEventListener('click', function() {
        if (!chart) return;
        var link = document.createElement('a');
        link.download = 'sensor_chart_' + new Date().toISOString().slice(0,10) + '.png';
        link.href = chart.toBase64Image('image/png');
        link.click();
    });

    initChart();
    initScrollDrag();
    fetchData();
    fetchHistory();
    setInterval(fetchData, 3000);
    setInterval(fetchHistory, 2000);
})();

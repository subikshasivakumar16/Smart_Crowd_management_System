/**
 * Crowd Monitoring Dashboard - Main JavaScript
 * Calendar, events, download, fullscreen, zone counts
 */

(function() {
    'use strict';

    const ZONES = ['A', 'B', 'C', 'D'];
    const MAX_PEOPLE = 50;
    const LOW_THRESH = 15;
    const MED_THRESH = 35;
    let selectedDate = new Date().toISOString().slice(0, 10);
    let calYear, calMonth;

    function getDensity(count) {
        const pct = (count / MAX_PEOPLE) * 100;
        if (pct <= LOW_THRESH) return { label: 'Low', cls: 'density-low' };
        if (pct <= MED_THRESH) return { label: 'Medium', cls: 'density-medium' };
        return { label: 'High', cls: 'density-high' };
    }

    function updateZone(zone, count) {
        const density = getDensity(count);
        const pct = Math.min(100, Math.round((count / MAX_PEOPLE) * 100));
        const ind = document.getElementById('ind-' + zone.toLowerCase());
        const densityEl = document.getElementById('density-' + zone.toLowerCase());
        const countEl = document.getElementById('count-' + zone.toLowerCase());
        const slider = document.getElementById('slider-' + zone.toLowerCase());
        if (ind) ind.className = 'density-badge ' + density.cls;
        if (densityEl) densityEl.textContent = density.label;
        if (countEl) countEl.textContent = count;
        if (slider) slider.value = pct;
    }

    function fetchCounts() {
        fetch('/counts')
            .then(r => r.json())
            .then(data => {
                let total = 0;
                ZONES.forEach(z => {
                    const c = data[z] !== undefined ? parseInt(data[z], 10) : 0;
                    total += c;
                    updateZone(z, c);
                });
                const el = document.getElementById('total-crowd');
                if (el) el.textContent = total;
            })
            .catch(() => ZONES.forEach(z => updateZone(z, 0)));
    }

    function renderCalendar() {
        const now = new Date();
        calYear = calYear || now.getFullYear();
        calMonth = calMonth || now.getMonth();
        const first = new Date(calYear, calMonth, 1);
        const last = new Date(calYear, calMonth + 1, 0);
        const startPad = first.getDay();
        const days = last.getDate();
        const monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        let html = '<div class="mb-2 d-flex justify-content-between align-items-center"><button class="btn btn-sm btn-outline-secondary" id="cal-prev">&laquo;</button><span class="fw-bold">' + monthNames[calMonth] + ' ' + calYear + '</span><button class="btn btn-sm btn-outline-secondary" id="cal-next">&raquo;</button></div>';
        html += '<div class="row g-1 text-center small fw-bold"><div class="col">Sun</div><div class="col">Mon</div><div class="col">Tue</div><div class="col">Wed</div><div class="col">Thu</div><div class="col">Fri</div><div class="col">Sat</div></div><div class="row g-1" id="cal-dates">';
        for (let i = 0; i < startPad; i++) html += '<div class="col date-cell text-muted"> </div>';
        fetch('/calendar/dates').then(r => r.json()).then(datesWithEvents => {
            for (let d = 1; d <= days; d++) {
                const dStr = calYear + '-' + String(calMonth + 1).padStart(2, '0') + '-' + String(d).padStart(2, '0');
                let cls = 'date-cell';
                if (datesWithEvents.includes(dStr)) cls += ' has-events';
                if (dStr === selectedDate) cls += ' selected';
                html += '<div class="col date-cell ' + cls + '" data-date="' + dStr + '">' + d + '</div>';
            }
            document.getElementById('calendar-body').innerHTML = html;
            document.querySelectorAll('#cal-dates .date-cell[data-date]').forEach(cell => {
                cell.addEventListener('click', () => {
                    selectedDate = cell.dataset.date;
                    document.querySelectorAll('.date-cell').forEach(c => c.classList.remove('selected'));
                    cell.classList.add('selected');
                    document.getElementById('sel-date').textContent = '(' + selectedDate + ')';
                    loadEvents(selectedDate);
                });
            });
            document.getElementById('cal-prev').onclick = () => { calMonth--; if (calMonth < 0) { calMonth = 11; calYear--; } renderCalendar(); };
            document.getElementById('cal-next').onclick = () => { calMonth++; if (calMonth > 11) { calMonth = 0; calYear++; } renderCalendar(); };
        }).catch(() => {});
    }

    function loadEvents(date) {
        fetch('/calendar/events?date=' + (date || selectedDate))
            .then(r => r.json())
            .then(events => {
                const el = document.getElementById('events-list');
                if (events.length === 0) {
                    el.innerHTML = '<p class="text-muted small mb-0">No events on this date</p>';
                } else {
                    el.innerHTML = events.reverse().map(e => '<div class="event-item ' + (e.type || '') + '"><strong>' + (e.time || '') + '</strong> ' + (e.message || '') + '</div>').join('');
                }
            })
            .catch(() => {});
    }

    document.getElementById('cal-today').addEventListener('click', () => {
        const t = new Date();
        selectedDate = t.toISOString().slice(0, 10);
        calYear = t.getFullYear();
        calMonth = t.getMonth();
        document.getElementById('sel-date').textContent = '(' + selectedDate + ')';
        loadEvents(selectedDate);
        renderCalendar();
    });

    document.querySelectorAll('.btn-fullscreen').forEach(btn => {
        btn.addEventListener('click', () => {
            const zone = btn.dataset.zone;
            const img = document.getElementById('video-' + zone.toLowerCase());
            const modal = document.getElementById('fullscreen-modal');
            const fsImg = document.getElementById('fullscreen-img');
            fsImg.src = img.src;
            modal.style.display = 'flex';
        });
    });
    document.getElementById('close-fullscreen').addEventListener('click', () => {
        document.getElementById('fullscreen-modal').style.display = 'none';
    });
    document.getElementById('fullscreen-modal').addEventListener('click', function(e) {
        if (e.target === this) this.style.display = 'none';
    });

    document.getElementById('sel-date').textContent = '(' + selectedDate + ')';
    renderCalendar();
    loadEvents(selectedDate);
    fetchCounts();
    setInterval(fetchCounts, 3000);
})();

// Shared page helpers for the Rosenhan and Dolos project pages (keep the two copies identical).
// Every string from data goes through esc() before it reaches innerHTML.
(function () {
  "use strict";

  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);
  const f2 = (v) => (v == null ? "n/a" : Number(v).toFixed(2));
  const clamp = (x) => Math.max(0, Math.min(100, x));

  // One fixed tooltip for the page. Any element with data-tip shows it on hover, tap or keyboard focus.
  function tooltip() {
    const tip = document.createElement("div");
    tip.id = "tip";
    tip.setAttribute("role", "tooltip");
    document.body.appendChild(tip);
    let cur = null;
    const place = (x, y) => {
      const r = tip.getBoundingClientRect();
      let left = x + 12, top = y + 12;
      if (left + r.width > innerWidth - 8) left = x - r.width - 12;
      if (top + r.height > innerHeight - 8) top = y - r.height - 12;
      tip.style.transform = `translate(${Math.max(8, left)}px, ${Math.max(8, top)}px)`;
    };
    const show = (el) => {
      if (el !== cur) { cur = el; tip.textContent = el.dataset.tip; }
      tip.classList.add("on");
    };
    const hide = () => { cur = null; tip.classList.remove("on"); };
    const onPointer = (e) => {
      const el = e.target.closest && e.target.closest("[data-tip]");
      if (!el) return hide();
      show(el);
      place(e.clientX, e.clientY);
    };
    document.addEventListener("pointermove", onPointer);
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("focusin", (e) => {
      const el = e.target.closest && e.target.closest("[data-tip]");
      if (!el) return hide();
      show(el);
      const r = el.getBoundingClientRect();
      place(r.left, r.bottom);
    });
    document.addEventListener("focusout", hide);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") hide(); });
    addEventListener("scroll", hide, { passive: true });
  }

  // Tabs with roving tabindex; arrows, Home and End move and select.
  function tabs(root, items, selected, onChange, label) {
    root.setAttribute("role", "tablist");
    if (label) root.setAttribute("aria-label", label);
    root.innerHTML = items.map((it) =>
      `<button type="button" role="tab" data-key="${esc(it.key)}">${esc(it.label)}</button>`).join("");
    const btns = [...root.children];
    const select = (key, focus) => {
      for (const b of btns) {
        const on = b.dataset.key === key;
        b.setAttribute("aria-selected", String(on));
        b.tabIndex = on ? 0 : -1;
        if (on && focus) b.focus();
      }
      onChange(key);
    };
    root.addEventListener("click", (e) => {
      const b = e.target.closest("[role=tab]");
      if (b) select(b.dataset.key, false);
    });
    root.addEventListener("keydown", (e) => {
      const i = btns.indexOf(document.activeElement);
      const j = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: btns.length - 1 }[e.key];
      if (i < 0 || j === undefined) return;
      e.preventDefault();
      select(btns[(j + btns.length) % btns.length].dataset.key, true);
    });
    select(selected, false);
    return { select };
  }

  // Grouped single-select chips (aria-pressed).
  function chips(root, groups, selected, onChange) {
    root.innerHTML = groups.map((g) =>
      `<div class="chipgroup" role="group" aria-label="${esc(g.name)}"><span>${esc(g.name)}</span>` +
      g.items.map((it) => `<button type="button" class="chip" data-key="${esc(it.key)}" aria-pressed="false">` +
        `${esc(it.label)}</button>`).join("") + "</div>").join("");
    const btns = [...root.querySelectorAll(".chip")];
    const select = (key) => {
      for (const b of btns) b.setAttribute("aria-pressed", String(b.dataset.key === key));
      onChange(key);
    };
    root.addEventListener("click", (e) => {
      const b = e.target.closest(".chip");
      if (b) select(b.dataset.key);
    });
    select(selected);
    return { select };
  }

  const ciText = (c) => (c.lo == null ? f2(c.v) : `${f2(c.v)} [${f2(c.lo)}, ${f2(c.hi)}]`);

  // Horizontal chart: one row per category, one track per series.
  // spec = {series:[{key,name,color,hollow?}], rows:[{label,sub?,tip?,cells:{key:{v,lo?,hi?,tip?}|{a,b,tip?}}}],
  //         mark:"bar"|"dot"|"dumbbell", domain:[min,max], ticks:[...], tickFmt, refs:[{x,label}], axis, ends?, table?}
  function chart(root, spec) {
    const [d0, d1] = spec.domain || [0, 1];
    const X = (v) => clamp(((v - d0) / (d1 - d0)) * 100);
    const tickFmt = spec.tickFmt || ((t) => String(t));
    const mark = spec.mark || "bar";
    const series = spec.series;
    const grid = (spec.ticks || []).map((t) => `<i class="gl" style="left:${X(t)}%"></i>`).join("") +
      (spec.refs || []).map((r) => `<i class="rl" style="left:${X(r.x)}%"></i>`).join("");
    const cellTip = spec.cellText || ((c) => (mark === "dumbbell" ? `${spec.ends[0]} ${f2(c.a)}, ${spec.ends[1]} ${f2(c.b)}` : ciText(c)));

    const track = (s, c, row) => {
      const tip = c.tip || `${row.label}\n${s.name}: ${cellTip(c)}`;
      let inner = "";
      if (mark === "dumbbell") {
        const lo = Math.min(c.a, c.b), hi = Math.max(c.a, c.b);
        inner = `<i class="link" style="left:${X(lo)}%;width:${X(hi) - X(lo)}%"></i>` +
          `<i class="dot" style="left:${X(c.a)}%"></i><i class="dot hollow" style="left:${X(c.b)}%"></i>`;
      } else {
        if (mark === "bar") {
          inner = c.v - d0 <= 0 ? `<i class="bar zero"></i>` : `<i class="bar" style="width:${X(c.v)}%"></i>`;
        }
        if (c.lo != null) inner += `<i class="ci" style="left:${X(c.lo)}%;width:${X(c.hi) - X(c.lo)}%"></i>`;
        if (mark === "dot") inner += `<i class="dot${c.hollow || s.hollow ? " hollow" : ""}" style="left:${X(c.v)}%"></i>`;
      }
      return `<div class="track${mark === "bar" ? "" : " tall"}" style="--c:${s.color}" data-tip="${esc(tip)}">${inner}</div>`;
    };

    let html = "";
    if (series.length > 1 || spec.refs || spec.ends) {
      html += '<ul class="legend">' + series.map((s) =>
        `<li><i class="sw${mark === "bar" ? "" : " round"}${s.hollow ? " hollow" : ""}" style="--c:${s.color}"></i>${esc(s.name)}</li>`).join("");
      if (spec.ends) {
        html += `<li><i class="sw round" style="--c:var(--muted)"></i>${esc(spec.ends[0])}</li>` +
          `<li><i class="sw hollow" style="--c:var(--muted)"></i>${esc(spec.ends[1])}</li>`;
      }
      html += (spec.refs || []).map((r) => `<li><i class="sw ref"></i>${esc(r.label)}</li>`).join("") + "</ul>";
    }
    for (const row of spec.rows) {
      const present = series.filter((s) => row.cells[s.key]);
      const rowTip = row.tip || [row.label, ...present.map((s) => `${s.name}: ${cellTip(row.cells[s.key])}`)].join("\n");
      html += `<div class="crow" tabindex="0" data-tip="${esc(rowTip)}"><div class="clabel">${esc(row.label)}` +
        (row.sub ? `<small>${esc(row.sub)}</small>` : "") + `</div><div class="plot">${grid}` +
        present.map((s) => track(s, row.cells[s.key], row)).join("") + "</div></div>";
    }
    html += '<div class="caxis" aria-hidden="true"><div></div><div class="ticks">' +
      (spec.ticks || []).map((t) => `<span style="left:${X(t)}%">${esc(tickFmt(t))}</span>`).join("") + "</div>" +
      (spec.axis ? `<div class="atitle">${esc(spec.axis)}</div>` : "") + "</div>";
    html += tableView(spec.table || {
      head: ["", ...series.map((s) => s.name)],
      rows: spec.rows.map((r) => [r.label, ...series.map((s) => (r.cells[s.key] ? cellTip(r.cells[s.key]) : ""))]),
    });
    root.classList.add("chart");
    root.innerHTML = html;
  }

  function tableView(t) {
    return '<details class="tview"><summary>Table view</summary><div class="table-scroll"><table class="data">' +
      (t.caption ? `<caption>${esc(t.caption)}</caption>` : "") +
      "<thead><tr>" + t.head.map((h, i) => `<th${i ? ' class="num"' : ""}>${esc(h)}</th>`).join("") + "</tr></thead><tbody>" +
      t.rows.map((r) => "<tr>" + r.map((c, i) => `<td${i ? ' class="num"' : ""}>${esc(c)}</td>`).join("") + "</tr>").join("") +
      "</tbody></table></div></details>";
  }

  // Slide viewer: buttons, arrow keys, counter. Only the current slide is requested.
  function carousel(root, slides) {
    let i = 0;
    root.classList.add("carousel");
    root.tabIndex = 0;
    root.setAttribute("aria-roledescription", "carousel");
    root.innerHTML = '<div class="stage"><img alt="" loading="lazy" decoding="async"></div><div class="cbar">' +
      '<button type="button" class="btn" data-d="-1" aria-label="Previous slide">Previous</button>' +
      '<span aria-live="polite"></span><button type="button" class="btn" data-d="1" aria-label="Next slide">Next</button></div>';
    const img = root.querySelector("img"), count = root.querySelector("span");
    const [prev, next] = root.querySelectorAll("button");
    const go = (j) => {
      i = Math.max(0, Math.min(slides.length - 1, j));
      img.src = slides[i].src;
      img.alt = slides[i].alt;
      count.textContent = `${i + 1} / ${slides.length}`;
      prev.disabled = i === 0;
      next.disabled = i === slides.length - 1;
    };
    root.addEventListener("click", (e) => {
      const b = e.target.closest("[data-d]");
      if (b) go(i + Number(b.dataset.d));
    });
    root.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") go(i - 1);
      else if (e.key === "ArrowRight") go(i + 1);
    });
    go(0);
  }

  function copyButton(btn, target) {
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(target.textContent);
        btn.textContent = "Copied";
      } catch (_) {
        const r = document.createRange();
        r.selectNodeContents(target);
        getSelection().removeAllRanges();
        getSelection().addRange(r);
        btn.textContent = "Selected";
      }
      setTimeout(() => { btn.textContent = "Copy"; }, 1500);
    });
  }

  // Marks the nav link of the section in view.
  function navSpy(nav) {
    const links = [...nav.querySelectorAll('a[href^="#"]')];
    const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    if (!("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const a = byId.get(e.target.id);
        if (!a) continue;
        for (const l of links) l.removeAttribute("aria-current");
        a.setAttribute("aria-current", "true");
        const box = a.parentElement;
        box.scrollTo({ left: a.offsetLeft - box.clientWidth / 2 + a.clientWidth / 2 });
      }
    }, { rootMargin: "-30% 0px -65% 0px" });
    for (const id of byId.keys()) {
      const s = document.getElementById(id);
      if (s) io.observe(s);
    }
  }

  // Replaces each [data-chart] fallback with a notice when site/data.js did not load.
  function requireData() {
    if (window.PAGE_DATA) return true;
    for (const el of document.querySelectorAll("[data-chart]")) {
      el.insertAdjacentHTML("beforeend", '<p class="nodata">Interactive view unavailable: site/data.js did not load.</p>');
    }
    return false;
  }

  function boot(run) {
    tooltip();
    const nav = document.querySelector(".toc");
    if (nav) navSpy(nav);
    for (const b of document.querySelectorAll("[data-copy]")) copyButton(b, document.getElementById(b.dataset.copy));
    if (requireData()) run(window.PAGE_DATA);
  }

  window.Lib = { esc, f2, tabs, chips, chart, tableView, carousel, boot };
})();

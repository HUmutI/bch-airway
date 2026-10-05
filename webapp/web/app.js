/* Two pages, one script. No framework: the app is an upload form and a poller. */
const AirwayApp = (() => {
  const $ = (id) => document.getElementById(id);
  const hex = (h) => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ---------------------------------------------------- percentiles for age
  // web/normative.json holds, per metric and two-year age band, the percentile
  // grid of the 90-scan normative cohort (built by tools/build_normative.py).
  // A measurement becomes a percentile by linear interpolation on that grid.
  let NORM = null, AGE = null;
  const loadNorm = async () => {
    if (NORM) return NORM;
    try { NORM = await (await fetch("/static/normative.json?v=17")).json(); } catch (_) { NORM = { metrics: {}, per_generation: {} }; }
    return NORM;
  };
  const bandOf = (age) => {
    if (age == null || !(age >= 0)) return null;
    const lo = Math.min(16, Math.floor(age / 2) * 2);
    return `${lo}-${lo + 2}`;
  };
  const percentile = (key, value, gen) => {
    if (!NORM || AGE == null || value == null || Number.isNaN(Number(value))) return null;
    const band = bandOf(AGE);
    const g = gen != null ? ((NORM.per_generation[key] || {})[String(gen)] || {})[band]
                          : (NORM.metrics[key] || {})[band];
    if (!g) return null;
    const v = Number(value), P = g.p, V = g.v, last = V.length - 1;
    let p, tag = "";
    if (v <= V[0]) { p = P[0]; tag = "<"; }
    else if (v >= V[last]) { p = P[last]; tag = ">"; }
    else {
      let i = 0; while (i < last && v > V[i + 1]) i++;
      p = P[i] + (P[i + 1] - P[i]) * (v - V[i]) / (V[i + 1] - V[i]);
    }
    return { p: Math.round(p), tag, n: g.n, ages: g.ages, extreme: p < 5 || p > 95 };
  };
  const pctBadge = (key, value, gen) => {
    if (AGE == null) return "";
    const r = percentile(key, value, gen);
    if (!r) return `<span class="pct none" title="no reference for this age band">–</span>`;
    return `<span class="pct${r.extreme ? " extreme" : ""}" title="percentile for age among ${r.n} healthy children aged ${r.ages} years">${r.tag}P${r.p}</span>`;
  };

  // ------------------------------------------------------------- upload page
  function upload() {
    const drop = $("drop"), input = $("file"), go = $("go"), ack = $("ack");
    let file = null;
    const ready = () => { go.disabled = !(file && ack.checked); };
    const pick = (f) => {
      if (!f) return;
      if (!/\.nii(\.gz)?$/i.test(f.name)) { $("err").textContent = "Choose a .nii or .nii.gz file."; return; }
      file = f; $("err").textContent = "";
      $("fname").textContent = `${f.name} · ${(f.size / 1048576).toFixed(1)} MB`;
      ready();
    };
    drop.onclick = () => input.click();
    input.onchange = () => pick(input.files[0]);
    ["dragenter", "dragover"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", ev => pick(ev.dataTransfer.files[0]));
    ack.onchange = ready;

    go.onclick = async () => {
      go.disabled = true; $("err").textContent = "";
      const fd = new FormData(); fd.append("file", file, file.name);
      if ($("age").value !== "") fd.append("age_years", $("age").value);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/jobs");
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) $("hint").textContent = `Uploading ${Math.round(100 * e.loaded / e.total)}%`;
      };
      xhr.onload = () => {
        if (xhr.status === 200) { location.href = `/jobs/${JSON.parse(xhr.responseText).id}`; return; }
        let m = "Upload failed."; try { m = JSON.parse(xhr.responseText).detail || m; } catch (_) {}
        $("err").textContent = m; $("hint").textContent = ""; go.disabled = false;
      };
      xhr.onerror = () => { $("err").textContent = "Network error."; go.disabled = false; };
      xhr.send(fd);
    };
  }

  // ---------------------------------------------------------------- job page
  function job() {
    const id = location.pathname.split("/").pop();
    $("jobid").textContent = `job ${id}`;
    let shown = false, morphShown = false, lastStatus = null;
    const fmt = (s) => s < 60 ? `${Math.round(s)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
    $("copy").onclick = async () => {
      try { await navigator.clipboard.writeText(location.href); $("copy").textContent = "Link copied"; }
      catch (_) { prompt("Copy this address:", location.href); }
      setTimeout(() => { $("copy").textContent = "Copy link"; }, 2000);
    };
    // patient age: from the upload, editable here; percentiles re-render on save
    $("agesave").onclick = async () => {
      const v = $("age").value;
      const r = await fetch(`/api/jobs/${id}/meta`, { method: "POST", headers: { "Content-Type": "application/json" },
                                                     body: JSON.stringify({ age_years: v === "" ? null : Number(v) }) });
      if (!r.ok) { $("agehint").textContent = "Age must be between 0 and 18 years."; return; }
      AGE = v === "" ? null : Number(v);
      $("agehint").textContent = AGE == null ? "No age given, so no percentiles." : `Percentiles for age ${AGE} y, against 90 healthy children (10 per two-year band).`;
      if (lastStatus && lastStatus.results_ready) renderStats(lastStatus.results || {});
      if (morphShown) showMorphometry(id);
    };
    const renderQC = (s, morph) => {
      const flags = [...(((s.results || {}).qc) || []), ...((morph && morph.qc) || [])];
      const box = $("qc");
      if (!s.results_ready) { box.hidden = true; return; }
      box.hidden = false;
      if (!flags.length) { box.className = "qc clean"; box.innerHTML = "<h3>Checks passed</h3>The airway reaches the lung apex, every lobe has branches, the tree has a normal size and depth, and no member mask had to be trimmed."; return; }
      const worst = flags.some(f => f.level === "warn") ? "warn" : "info";
      box.className = `qc ${worst}`;
      box.innerHTML = `<h3>${worst === "warn" ? "Check these before trusting the result" : "Notes on this result"}</h3><ul>` +
        flags.map(f => `<li class="${f.level}">${esc(f.text)}</li>`).join("") + "</ul>";
    };

    async function tick() {
      let s;
      try { s = await (await fetch(`/api/jobs/${id}`, { cache: "no-store" })).json(); }
      catch (_) { setTimeout(tick, 4000); return; }
      const p = Math.round((s.progress || 0) * 100);
      $("bar").querySelector("i").style.width = `${s.state === "done" ? 100 : p}%`;
      $("pct").textContent = s.state === "done" ? "100%" : `${p}%`;
      $("msg").textContent = s.message || "";
      $("bar").className = "bar" + (s.state === "done" ? " done" : s.state === "failed" ? " failed" : "");
      $("title").textContent = { queued: "Queued", claimed: "Starting", running: "Running",
                                 done: "Finished", failed: "Failed" }[s.state] || s.state;
      const now = Date.now() / 1000;
      $("stages").innerHTML = (s.stages || []).map(st => {
        const t = st.started ? fmt((st.ended || now) - st.started) : "";
        return `<li class="${st.state}"><i></i>${st.label}<span class="t">${t}</span></li>`;
      }).join("");
      $("err").textContent = s.error || "";
      $("failrow").hidden = s.state !== "failed";
      try { $("log").textContent = await (await fetch(`/api/jobs/${id}/log`)).text(); } catch (_) {}

      // The viewer opens as soon as the segmentation is packaged. Morphometry
      // is still running behind it and fills its own card when it is done.
      lastStatus = s;
      if (s.age_years != null && AGE == null && !shown) { AGE = s.age_years; $("age").value = s.age_years; }
      if (s.results_ready && !shown) {
        shown = true;
        await loadNorm();
        $("agehint").textContent = AGE == null ? "Enter the patient's age to see every measurement as a percentile for age." : `Percentiles for age ${AGE} y, against 90 healthy children (10 per two-year band).`;
        showResults(id, s.results || {});
      }
      if (shown) renderQC(s, window._morph);
      const mst = (s.stages || []).find(x => x.name === "morphometry");
      if (shown && mst && !morphShown) {
        $("morph").hidden = false;
        if (mst.state === "done") { morphShown = true; showMorphometry(id).then(() => renderQC(s, window._morph)); }
        else if (mst.state === "failed") {
          morphShown = true;
          $("morph-msg").textContent = "Measurement failed: " + (s.morphometry_error || "see log");
        } else if (mst.state === "running") {
          $("morph-msg").textContent = `Measuring the tree (${fmt(now - mst.started)}), the viewer above is already usable.`;
        }
      }
      if (s.state === "done") return;
      if (s.state !== "failed") setTimeout(tick, 2500);
    }
    $("retry").onclick = async () => {
      $("retry").disabled = true;
      const r = await fetch(`/api/jobs/${id}/retry`, { method: "POST" });
      if (r.ok) { $("results").hidden = true; $("morph").hidden = true; shown = morphShown = false;
                  $("retry").disabled = false; tick(); }
      else { $("err").textContent = "Could not retry."; $("retry").disabled = false; }
    };
    tick();
  }

  // --------------------------------------------------------------- results
  function renderStats(r) {
    const stats = [];
    if (r.airway_volume_ml != null) stats.push([`${r.airway_volume_ml} mL`, "airway lumen volume", pctBadge("airway_lumen_volume_ml", r.airway_volume_ml)]);
    if (r.branches != null) stats.push([r.branches, "branches named", pctBadge("branch_count", r.branches)]);
    if (r.lobes) stats.push([r.lobes.filter(l => l.name !== "trachea").length, "lobes reached", ""]);
    if (r.mean_confidence_pct != null) stats.push([`${Math.round(r.mean_confidence_pct)} %`, "mean confidence", ""]);
    if (r.view_spacing_mm) stats.push([`${r.view_spacing_mm} mm`, "display resolution", ""]);
    if (r.source_orientation) stats.push([r.source_orientation, "output orientation", ""]);
    $("stats").innerHTML = stats.map(([b, s, p]) => `<div class="stat"><b>${b}</b><span>${s}</span>${p}</div>`).join("");
  }

  async function showResults(id, r) {
    $("results").hidden = false;
    const f = (n) => `/api/jobs/${id}/files/${n}`;
    renderStats(r);
    const dl = [
      ["airway_results.zip", "Everything (zip)", "zip"],
      ["itksnap_bundle.zip", "ITK-SNAP bundle (labels + label descriptions + how-to)", "zip"],
      ["airway_mask.nii.gz", "Airway mask"], ["airway_lobar.nii.gz", "Lobar labels"],
      ["airway_segmental.nii.gz", "Segmental labels"], ["airway_branches.nii.gz", "Branch instances"],
      ["airway_centreline.nii.gz", "Centreline"], ["airway_confidence.nii.gz", "Confidence map"],
      ["mesh_airway.mz3", "Airway surface (MZ3)"],
      ["itksnap_lobar_labels.txt", "ITK-SNAP lobar labels"], ["itksnap_segmental_labels.txt", "ITK-SNAP segmental labels"],
      ["labels.json", "Branch names (JSON)"], ["summary.json", "Summary (JSON)"],
    ];
    $("dl").innerHTML = dl.map(([n, t, c]) => `<a class="${c || ""}" href="${f(n)}" download>${t}</a>`).join("");

    let labels = {}, pal = {};
    try { labels = await (await fetch(f("labels.json"))).json(); } catch (_) {}
    try { pal = await (await fetch("/static/palette.json")).json(); } catch (_) {}
    const cat = pal.categorical || ["#d62b2b", "#eda100", "#1baf7a", "#e87ba4", "#8e5aa8", "#eb6834"];
    const lobNames = labels.lobar || {}, segNames = labels.segmental || {};
    const lobColours = {}, segColours = {}, brColours = {}, brNames = {};
    Object.keys(lobNames).forEach((v, i) => { lobColours[v] = (pal.lobar || {})[lobNames[v]] || cat[i % cat.length]; });
    Object.keys(segNames).forEach((v, i) => { segColours[v] = cat[i % cat.length]; });
    Object.keys(labels.branches || {}).forEach((v, i) => {
      brColours[v] = cat[i % cat.length];
      const b = labels.branches[v];
      brNames[v] = Array.isArray(b) ? b.filter((x, j, a) => a.indexOf(x) === j).join(" · ") : String(b);
    });
    const airwayColour = pal.airway || "#d62b2b";
    window._labels = labels;                 // the branch table reads branch_confidence from here
    const boot = () => viewer(id, f, { lobNames, lobColours, segNames, segColours, brColours, brNames,
                                        airwayColour, centreColour: pal.centreline || "#f5e663",
                                        meshes: r.meshes || [],
                                        confidence: { mean: r.mean_confidence_pct, low: r.low_confidence_fraction_pct } });
    if (window.Niivue) boot(); else window.addEventListener("niivue-ready", boot, { once: true });
  }

  function labelColormap(colours) {
    // Discrete label LUT for NVImage.setColormapLabel: value i -> colour i, 0 transparent.
    const keys = Object.keys(colours).map(Number);
    const n = (keys.length ? Math.max(...keys) : 0) + 1;
    const R = [], G = [], B = [], A = [], I = [], L = [];
    for (let i = 0; i < n; i++) {
      const c = colours[i] ? hex(colours[i]) : [0, 0, 0];
      R.push(c[0]); G.push(c[1]); B.push(c[2]); A.push(colours[i] ? 255 : 0); I.push(i); L.push(String(i));
    }
    return { R, G, B, A, I, labels: L };
  }

  // Confidence ramp, the same stops stage_package.py paints the surface with:
  // 0 % red, 50 % yellow, 100 % green. NiiVue builds a 256-entry LUT by
  // interpolating between the I stops; entry 0 has alpha 0 so everything at or
  // below cal_min (the 0 outside the mask) is transparent, and the overlay
  // shader turns any non-zero alpha opaque, so the rest is solid colour.
  const CONF_STOPS = [[0, [214, 39, 40]], [50, [255, 200, 0]], [100, [44, 160, 44]]];
  const CONF_CMAP = { R: [214, 214, 255, 44], G: [39, 39, 200, 160], B: [40, 40, 0, 44], A: [0, 255, 255, 255], I: [0, 1, 128, 255] };
  const confColour = (pct) => {
    pct = Math.min(100, Math.max(0, Number(pct) || 0));
    for (let i = 0; i < CONF_STOPS.length - 1; i++) {
      const [p0, c0] = CONF_STOPS[i], [p1, c1] = CONF_STOPS[i + 1];
      if (pct <= p1) { const t = (pct - p0) / (p1 - p0); return `rgb(${c0.map((a, k) => Math.round(a + t * (c1[k] - a))).join(",")})`; }
    }
    return `rgb(${CONF_STOPS[CONF_STOPS.length - 1][1].join(",")})`;
  };
  const confCell = (pct) => pct == null ? "–" : `<span class="conf"><i style="background:${confColour(pct)}"></i>${Math.round(pct)} %</span>`;

  async function viewer(id, f, L) {
    const say = (t) => { $("readout").textContent = t; };
    const fail = (where, e) => { console.error(where, e); say(`Viewer error in ${where}: ${e && e.message || e}`); };
    window.addEventListener("unhandledrejection", (ev) => fail("promise", ev.reason));

    const nv = new window.Niivue({
      sliceType: 3, multiplanarLayout: 2, multiplanarShowRender: 1, multiplanarPadPixels: 3,
      isSliceMM: true, show3Dcrosshair: true, isOrientCube: true, isRadiologicalConvention: true,
      isHighResolutionCapable: false, isColorbar: false, dragMode: 1,
      crosshairColor: [0.93, 0.63, 0.0, 1], crosshairWidth: 1, backColor: [0, 0, 0, 1],
      meshThicknessOn2D: 0, meshXRay: 0, textHeight: 0.04, loadingText: "loading",
    });
    try { await nv.attachTo("gl"); } catch (e) { fail("WebGL", e); return; }
    window.airwayViewer = nv;
    const NVImage = window.NVImage;
    try { nv.addColormap("confidence", CONF_CMAP); } catch (e) { console.warn("confidence colormap", e); }

    // ---- state ---------------------------------------------------------------
    const OVERLAYS = {
      airway:    { file: "airway_mask_view.nii.gz",      colours: { 1: L.airwayColour }, names: { 1: "airway" } },
      lobar:     { file: "airway_lobar_view.nii.gz",     colours: L.lobColours, names: L.lobNames },
      segmental: { file: "airway_segmental_view.nii.gz", colours: L.segColours, names: L.segNames },
      branches:  { file: "airway_branches_view.nii.gz",  colours: L.brColours, names: L.brNames },
      // continuous: a percent per voxel, not a label; see CONF_CMAP
      confidence: { file: "airway_confidence_view.nii.gz", continuous: true, colormap: "confidence" },
    };
    const cache = {};
    let ct = null, overlay = null, overlayKey = null, centre = null, layout = "grid";
    const lobeOn = {};
    L.meshes.filter(m => m.kind === "lobe").forEach(m => { lobeOn[m.label] = true; });

    // ---- helpers --------------------------------------------------------------
    const load = async (key) => {
      if (cache[key]) return cache[key];
      const o = OVERLAYS[key] || { file: "airway_centreline_view.nii.gz", colours: { 1: L.centreColour } };
      // the name must keep its extension: NiiVue picks the reader from it
      const img = await NVImage.loadFromUrl({ url: f(o.file), name: o.file, colormap: o.continuous ? o.colormap : "gray", opacity: 1 });
      if (o.continuous) { img.cal_min = 1; img.cal_max = 100; }      // 0 = outside the mask, transparent
      else img.setColormapLabel(labelColormap(o.colours));
      cache[key] = img;
      return img;
    };
    const showWL = () => { if (ct) $("wl").textContent = `W ${Math.round(ct.cal_max - ct.cal_min)} · L ${Math.round((ct.cal_max + ct.cal_min) / 2)} HU`; };
    const setWindow = (lo, hi) => { if (!ct) return; ct.cal_min = lo; ct.cal_max = hi; nv.updateGLVolume(); showWL(); };
    const opacity = () => $("ov-op").value / 100;
    // NiiVue reads the mesh format from the name, so names keep their extension
    const meshName = (n) => `${n}.mz3`;
    const meshBase = (m) => m.name.replace(/\.mz3$/i, "");
    const meshByName = (n) => nv.meshes.find(m => m.name === meshName(n));
    const slicesIn3D = () => layout !== "4" || $("ck-sl3d").checked;

    // In the 3D-only layout the default is the surface alone. The slice
    // shader ignores volume opacity, so hiding the planes means switching the
    // 3D pane to the ordinary volume shader with a fully transparent volume
    // and no overlays. Switching shaders re-uploads the CT, so only do it on
    // a change.
    let planesMode = null;
    const applyVisibility = async () => {
      const planes = slicesIn3D();
      if (planes !== planesMode) {
        planesMode = planes;
        try {
          await nv.setVolumeRenderIllumination(planes ? -1 : 0);
        } catch (e) { console.warn("render shader", e); }
        nv.opts.renderOverlayBlend = planes ? 1 : 0;
      }
      // Every call, not only on a mode change: over a slow link the user can
      // press "3D" before the CT has arrived, and the opacity set here must
      // still land once it has (otherwise the CT hides the surface inside it).
      if (ct && nv.volumes.includes(ct)) nv.setOpacity(0, planes ? 1 : 0);
      if (overlay && nv.volumes.includes(overlay)) nv.setOpacity(nv.volumes.indexOf(overlay), opacity());
      nv.opts.show3Dcrosshair = planes && $("ck-xh").checked;
      nv.drawScene();
    };
    // Which surface is shown follows the overlay radio: the plain red airway,
    // the per-lobe surfaces, or the airway coloured by segment / branch
    // (mesh_airway_segmental.mz3, mesh_airway_branches.mz3, written by the
    // package stage with one RGBA per vertex). Older results lack the
    // coloured ones and fall back to the red surface.
    const hasMesh = (n) => !!meshByName(n);
    const showMeshes = () => {
      const on = $("ck-mesh").checked, op = $("mesh-op").value / 100;
      const want = overlayKey === "lobar" ? "lobes"
                 : ["segmental", "branches", "confidence"].includes(overlayKey) && hasMesh(`airway_${overlayKey}`) ? `airway_${overlayKey}`
                 : "airway";
      for (const m of nv.meshes) {
        const base = meshBase(m);
        const vis = on && (want === "lobes" ? (base.startsWith("lobe_") && !!lobeOn[+base.split("_")[1]]) : base === want);
        // set unconditionally: the cached flags can disagree with the GL state
        // after meshes were added while the scene was being rebuilt
        nv.setMeshProperty(m.id, "visible", vis);
        nv.setMeshProperty(m.id, "opacity", op);
      }
      window._meshState = nv.meshes.map(m => [m.name, m.visible, m.opacity, m.indexCount]);
      nv.drawScene();
    };
    const legend = (key) => {
      const o = OVERLAYS[key];
      if (o.continuous) {
        const c = L.confidence || {};
        const grad = CONF_STOPS.map(([p, rgb]) => `rgb(${rgb.join(",")}) ${p}%`).join(", ");
        $("legend").innerHTML =
          `<div class="ramp" style="background:linear-gradient(90deg, ${grad})"></div>` +
          `<span class="ramp-lab"><b>0 %</b> one model, likely wrong</span>` +
          `<span class="ramp-lab"><b>50 %</b> two models, or a split AirMorph vote</span>` +
          `<span class="ramp-lab"><b>100 %</b> all models agree</span>` +
          (c.mean != null ? `<span class="ramp-stat">mean ${Math.round(c.mean)} % · ${c.low != null ? Math.round(c.low) : "–"} % of the mask below 50 %</span>` : "") +
          `<span class="more">P(airway) given which models marked the voxel, calibrated on ATM'22.</span>`;
        return;
      }
      const keys = Object.keys(o.colours);
      const rows = keys.slice(0, 40).map(k => {
        const box = key === "lobar" && meshByName(`lobe_${k}`)
          ? `<input type="checkbox" data-lobe="${k}" ${lobeOn[k] ? "checked" : ""} title="show in 3D">` : "";
        return `<span><i style="background:${o.colours[k]}"></i>${esc(o.names[k] || k)}${box}</span>`;
      });
      if (keys.length > 40) rows.push(`<span class="more">and ${keys.length - 40} more</span>`);
      $("legend").innerHTML = rows.join("");
      $("legend").querySelectorAll("input[data-lobe]").forEach(cb => cb.onchange = () => { lobeOn[cb.dataset.lobe] = cb.checked; showMeshes(); });
    };
    const setOverlay = async (key) => {
      try {
        say(`loading ${key} labels…`);
        const img = await load(key);
        if (overlay && nv.volumes.includes(overlay)) nv.removeVolume(overlay);
        overlay = img; overlayKey = key;
        img.opacity = opacity();
        nv.addVolume(img);
        if (centre && nv.volumes.includes(centre)) { nv.removeVolume(centre); nv.addVolume(centre); }
        legend(key);
        showMeshes();
        applyVisibility();
        say("Click a slice to read position and value");
      } catch (e) {
        if (key === "confidence") { say("This result has no confidence map (it was run before the map existed). Re-run the scan to get one."); return; }
        fail(`${key} overlay`, e);
      }
    };
    const setLayout = (v) => {
      layout = v;
      if (v === "grid") { nv.opts.multiplanarShowRender = 1; nv.setSliceType(3); }
      else nv.setSliceType(Number(v));
      planesMode = null;                     // re-establish the render state, whatever happened meanwhile
      $("layouts").querySelectorAll("button").forEach(x => x.classList.toggle("on", x.dataset.layout === v));
      $("ck-sl3d").parentElement.style.opacity = v === "4" ? 1 : 0.5;
      applyVisibility();
    };

    // ---- controls: wired before anything loads, so they always respond ---------
    $("layouts").querySelectorAll("button").forEach(b => b.onclick = () => setLayout(b.dataset.layout));
    $("presets").querySelectorAll("button").forEach(b => b.onclick = () => {
      const [lo, hi] = b.dataset.w.split(",").map(Number); setWindow(lo, hi);
      $("presets").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    });
    $("ov-op").oninput = applyVisibility;
    document.querySelectorAll('input[name="ov"]').forEach(r => r.onchange = () => r.checked && setOverlay(r.value));
    $("ck-cl").onchange = async () => {
      try {
        if ($("ck-cl").checked) { centre = await load("centreline"); centre.opacity = 1; nv.addVolume(centre); applyVisibility(); }
        else if (centre && nv.volumes.includes(centre)) nv.removeVolume(centre);
      } catch (e) { fail("centreline", e); }
    };
    $("ck-mesh").onchange = showMeshes;
    $("mesh-op").oninput = showMeshes;
    $("ck-sl3d").onchange = applyVisibility;
    $("ck-xh").onchange = () => { nv.setCrosshairWidth($("ck-xh").checked ? 1 : 0); applyVisibility(); };
    $("shot").onclick = () => nv.saveScene(`airway_${id}.png`).catch(e => fail("screenshot", e));
    nv.onIntensityChange = () => { showWL(); $("presets").querySelectorAll("button").forEach(b => b.classList.remove("on")); };

    // ---- readout ------------------------------------------------------------------
    // The label under the crosshair, searching a small neighbourhood because
    // the surface (and a click on it) sits on the boundary between the label
    // and the background.
    const labelAt = (img, v, radius = 2) => {
      const x0 = Math.round(v[0]), y0 = Math.round(v[1]), z0 = Math.round(v[2]);
      const get = (x, y, z) => { try { return img.getValue(x, y, z) || 0; } catch (_) { return 0; } };
      let val = get(x0, y0, z0);
      if (val > 0) return val;
      for (let r = 1; r <= radius; r++) {
        const counts = {};
        for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) for (let dz = -r; dz <= r; dz++) {
          const t = get(x0 + dx, y0 + dy, z0 + dz);
          if (t > 0) counts[t] = (counts[t] || 0) + 1;
        }
        const keys = Object.keys(counts);
        if (keys.length) return +keys.reduce((a, b) => counts[a] >= counts[b] ? a : b);
      }
      return 0;
    };
    let branchesImg = null;
    nv.onLocationChange = (d) => {
      try {
        const v = d.vox, mm = d.mm;
        const hu = d.values && d.values[0] ? Math.round(d.values[0].value) : "";
        let br = "", brId = 0;
        if (branchesImg && v) {
          brId = labelAt(branchesImg, v);
          if (brId > 0) br = ` · ${L.brNames[String(brId)] || `branch ${brId}`} (#${brId})`;
        }
        let cf = "";
        if (overlayKey === "confidence" && overlay && v) {
          const pct = labelAt(overlay, v, 1);
          if (pct > 0) cf = ` · confidence ${pct} %`;
        }
        say(`voxel ${Array.from(v).map(Math.round).join(", ")} · ` +
            `${Array.from(mm).slice(0, 3).map(x => x.toFixed(1)).join(", ")} mm · ${hu} HU${br}${cf}`);
      } catch (_) {}
    };

    // ---- load, smallest first, so the airway is on screen within seconds ----
    // Order: the airway surface (a few MB), a coarse CT (a few MB) with the
    // airway labels on it, then the full-resolution CT (tens of MB) which
    // replaces the coarse one in place. Over a slow link the old order (CT
    // first) left the page looking empty for the whole download.
    const busy = (text) => {
      const el = $("loading");
      if (!el) return;
      if (!text) { el.hidden = true; return; }
      el.hidden = false; el.textContent = text;
    };
    const loadCT = (name) => NVImage.loadFromUrl({ url: f(name), name, colormap: "gray", opacity: 1 });
    const useCT = async (img) => {
      const old = ct;
      nv.addVolume(img);
      if (nv.volumes[0] !== img && typeof nv.setVolume === "function") nv.setVolume(img, 0);
      if (old && nv.volumes.includes(old)) nv.removeVolume(old);
      ct = img;
      setWindow(-1000, 300);
      // adding or removing a volume resets NiiVue's render shader; the 3D
      // mode's "transparent CT + surface" state has to be set up again
      planesMode = null;
      await applyVisibility();
      showMeshes();
    };

    // 1. the surface, visible on its own in the 3D pane
    const meshList = [{ url: f("mesh_airway.mz3"), name: meshName("airway"), rgba255: [...hex(L.airwayColour), 255], visible: true }];
    for (const m of L.meshes.filter(m => m.kind === "lobe")) {
      meshList.push({ url: f(m.file), name: meshName(`lobe_${m.label}`),
                      rgba255: [...hex(L.lobColours[String(m.label)] || "#999999"), 255], visible: false });
    }
    let haveMesh = false;
    try {
      busy("Loading the airway surface…"); say("loading the airway surface…");
      await nv.loadMeshes(meshList);
      haveMesh = true;
      showMeshes();
      nv.setSliceType(4);                    // nothing but the surface yet: show it in 3D
    } catch (e) { fail("surfaces", e); }
    // the label-coloured copies of the surface, each optional
    const NVMesh = window.NVMesh;
    const loadColoured = async () => {
      if (!NVMesh || !NVMesh.loadFromUrl) return;
      for (const key of ["segmental", "branches", "confidence"]) {
        try {
          const m = await NVMesh.loadFromUrl({ url: f(`mesh_airway_${key}.mz3`), gl: nv.gl, name: meshName(`airway_${key}`), visible: false });
          nv.addMesh(m);
        } catch (_) { /* older result without coloured surfaces */ }
      }
      showMeshes();
    };

    // 2. a coarse CT quickly (older results do not have it), then the labels
    let low = null;
    try { busy("Loading a preview of the CT…"); say("loading a preview of the CT…"); low = await loadCT("image_view_lowres.nii.gz"); }
    catch (_) { low = null; }
    if (low) {
      await useCT(low);
      setLayout(layout);
      await applyVisibility();
      await setOverlay("airway");
      if (haveMesh) legend(overlayKey);
    }

    // 3. the full-resolution CT, swapped in underneath the labels
    try {
      busy(low ? "Loading the full-resolution CT in the background…" : "Loading the CT…");
      say("loading the full-resolution CT…");
      const full = await loadCT("image_view.nii.gz");
      await useCT(full);
      if (!low) { setLayout(layout); await applyVisibility(); await setOverlay("airway"); if (haveMesh) legend(overlayKey); }
      else await applyVisibility();
    } catch (e) {
      if (!low) { busy(null); fail("CT", e); return; }
      console.warn("full-resolution CT not loaded, keeping the preview", e);
    }
    busy(null);
    say("Click a slice to read position and value. Left-drag rotates the 3D view.");
    loadColoured().catch(() => {});
    load("branches").then(img => { branchesImg = img; window._branchesImg = img; }).catch(() => {});
  }

  // ------------------------------------------------------------- morphometry
  const num = (x, d = 1) => (x == null || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
  const table = (head, rows, cls = "") =>
    `<table class="m ${cls}"><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;

  async function showMorphometry(id) {
    const f = (n) => `/api/jobs/${id}/files/${n}`;
    let m;
    try { m = await (await fetch(f("morphometry.json"))).json(); }
    catch (_) { $("morph-msg").textContent = "Measurements are not available for this scan."; return; }
    const c = m.central || {}, x = c.cross_section || {};
    const ok = c.status === "ok", xok = ok && x.trachea_mean_diameter_mm != null;
    window._morph = m;
    $("morph-msg").textContent = "";
    const P = (key, v, gen) => pctBadge(key, v, gen);

    const stats = [
      [m.branch_count, "branches", P("branch_count", m.branch_count)],
      [`${num(m.total_centreline_length_mm, 0)} mm`, "total centreline length", P("total_centreline_length_mm", m.total_centreline_length_mm)],
      [m.max_generation, "deepest generation", P("max_generation", m.max_generation)],
      [m.airway_lumen_volume_ml != null ? `${num(m.airway_lumen_volume_ml)} mL` : "–", "lumen volume", P("airway_lumen_volume_ml", m.airway_lumen_volume_ml)],
      [xok ? `${num(x.trachea_length_intrathoracic_mm)} mm` : ok ? `${num(c.trachea.length_mm)} mm` : "–",
       xok ? "intrathoracic tracheal length" : "tracheal length", xok ? P("trachea_length_intrathoracic_mm", x.trachea_length_intrathoracic_mm) : ""],
      [xok ? `${num(x.trachea_mean_diameter_mm)} mm` : ok ? `${num(c.trachea.diameter_edt_mean_mm)} mm` : "–",
       xok ? "tracheal diameter (cross section)" : "tracheal diameter (inscribed)", xok ? P("trachea_mean_diameter_mm", x.trachea_mean_diameter_mm) : ""],
      [xok && x.airway_to_lung_volume_percent != null ? `${num(x.airway_to_lung_volume_percent, 2)} %` : "–", "airway / lung volume", xok ? P("airway_to_lung_volume_percent", x.airway_to_lung_volume_percent) : ""],
      [m.skeleton_fractal_dimension != null ? num(m.skeleton_fractal_dimension, 2) : "–", "skeleton fractal dimension", P("skeleton_fractal_dimension", m.skeleton_fractal_dimension)],
    ];
    let html = `<div class="stats">${stats.map(([b, s, p]) => `<div class="stat"><b>${b}</b><span>${s}</span>${p}</div>`).join("")}</div>`;
    if (AGE == null) html += `<p class="mnote">Enter the patient's age above to see each value as a percentile for age.</p>`;
    html += `<div class="mgrid">`;

    // central airways
    if (ok) {
      const row = (name, b, dx, lx, kL, kD) => [name,
        num(lx != null ? lx : b.length_mm) + (lx != null && kL ? P(kL, lx) : ""),
        num(b.diameter_edt_mean_mm),
        dx != null ? num(dx) + (kD ? P(kD, dx) : "") : "–",
        b.angle_deg != null ? num(b.angle_deg, 0) : "–"];
      let rows = [row("Trachea", c.trachea, x.trachea_mean_diameter_mm, x.trachea_length_intrathoracic_mm, "trachea_length_intrathoracic_mm", "trachea_mean_diameter_mm"),
                  row("Right main bronchus", c.rmb, x.rmb_mean_diameter_mm, x.rmb_length_mm, "rmb_length_mm", "rmb_mean_diameter_mm"),
                  row("Left main bronchus", c.lmb, x.lmb_mean_diameter_mm, x.lmb_length_mm, "lmb_length_mm", "lmb_mean_diameter_mm")];
      html += `<div><h3>Central airways</h3>${table(["", "Length mm", "Ø inscribed mm", "Ø cross section mm", "Angle °"], rows)}`;
      if (xok) {
        const ratios = [
          ["Tracheal short / long axis", `${num(x.trachea_short_axis_mm)}${P("trachea_short_axis_mm", x.trachea_short_axis_mm)} / ${num(x.trachea_long_axis_mm)}${P("trachea_long_axis_mm", x.trachea_long_axis_mm)} mm`],
          ["Tracheal eccentricity (short / long)", num(x.trachea_eccentricity, 2) + P("trachea_eccentricity", x.trachea_eccentricity)],
          ["Tracheal taper (distal / proximal Ø)", num(x.trachea_taper_distal_over_proximal, 2) + P("trachea_taper_distal_over_proximal", x.trachea_taper_distal_over_proximal)],
          ["Tracheal length / diameter", num(x.trachea_length_over_diameter, 1) + P("trachea_length_over_diameter", x.trachea_length_over_diameter)],
          ["Daughter / parent Ø at carina", num(x.daughter_over_parent_diameter, 2) + P("daughter_over_parent_diameter", x.daughter_over_parent_diameter)],
          ["Right-left asymmetry (R−L)/(R+L)", num(x.right_left_asymmetry, 2) + P("right_left_asymmetry", x.right_left_asymmetry)],
          ["Interbronchial angle (proxy, sum of local angles)", c.interbronchial_angle_proxy_deg != null ? `${num(c.interbronchial_angle_proxy_deg, 0)} °` : "–"],
          ["Segmented lung volume", `${num(x.lung_volume_ml, 0)} mL` + P("lung_volume_ml", x.lung_volume_ml)],
        ];
        html += table(["Ratio", ""], ratios);
      } else if (x.reason) {
        html += `<p class="mnote">Cross sections not available: ${esc(x.reason)}</p>`;
      }
      html += `</div>`;
    } else {
      html += `<div><h3>Central airways</h3><p class="mnote">Not available: ${esc(c.reason || "tree too sparse")}</p></div>`;
    }
    // per generation
    html += `<div><h3>Per generation</h3>${table(["Gen", "n", "Ø mean mm", "Length mean mm", "Length total mm", "Angle °"],
      (m.per_generation || []).map(g => [g.generation, g.n + P("n", g.n, g.generation),
                                         num(g.diameter_mean_mm) + P("diameter_mean_mm", g.diameter_mean_mm, g.generation),
                                         num(g.length_mean_mm) + P("length_mean_mm", g.length_mean_mm, g.generation),
                                         num(g.length_total_mm, 0), g.angle_mean_deg != null ? num(g.angle_mean_deg, 0) : "–"]))}</div>`;
    // per lobe
    html += `<div><h3>Per lobe</h3>${table(["Lobe", "n", "Deepest gen", "Length mm", "Volume mL", "Ø mean mm"],
      (m.per_lobe || []).map(l => [esc(l.lobe), l.n, l.max_generation, num(l.length_total_mm, 0), num(l.volume_ml, 2), num(l.diameter_mean_mm)]))}</div>`;
    html += `</div>`;

    // every branch
    let rows = [];
    try { rows = await (await fetch(f("morphometry_table.json"))).json(); } catch (_) {}
    if (rows.length) {
      rows.sort((a, b) => a.generation - b.generation || a.branch_instance_id - b.branch_instance_id);
      // mean confidence per branch (labels.json, written by the package stage); older results lack it
      let bc = (window._labels || {}).branch_confidence;
      if (!bc) { try { bc = (await (await fetch(f("labels.json"))).json()).branch_confidence; } catch (_) {} }
      const conf = bc ? (r) => [confCell(bc[String(r.branch_instance_id)])] : () => [];
      html += `<h3 style="font-size:13px;margin:18px 0 6px">All branches <span class="hint">(click a row to jump there)</span></h3><div class="mwrap">` +
        table(["#", "Gen", "Name", "Lobe", "Segment", ...(bc ? ["Conf"] : []), "Length mm", "Ø mean mm", "Ø min mm", "Tortuosity", "Angle °", "Volume mm³", "QC"],
          rows.map(r => [r.branch_instance_id, r.generation, esc(r.branch_name), esc(r.lobar_name), esc(r.segmental_name), ...conf(r),
            num(r.centerline_length_mm), num(r.diameter_mean_mm), num(r.diameter_min_mm), num(r.tortuosity_length_ratio, 2),
            r.branch_angle_deg != null ? num(r.branch_angle_deg, 0) : "–", num(r.volume_mm3, 0), esc(r.geometry_qc_flags || "")])) + `</div>`;
    }
    const bcNote = html.includes(">Conf<") ? '"Conf" is the branch\'s mean voxel confidence: the probability that it is airway given which of the three models marked it, calibrated on ATM\'22; below 50 % means one model alone. ' : "";
    html += `<p class="mnote">${AGE != null ? "<b>P</b> badges are percentiles for age: P36 means larger than in 36 % of healthy children of the same two-year age band (about 30 reference scans). &lt;P3 and &gt;P98 are outside what the reference can resolve; below P5 and above P95 are highlighted. Rows without a badge have no reference (e.g. generations beyond 10). " : ""}Measured on the segmented lumen with the centreline of the fused mask. Diameters:
      "inscribed" is twice the distance-transform radius along the centreline, "cross section" is the mean of short and
      long axis of the lumen in planes perpendicular to the centreline. Angles are between a branch and its parent.
      ${bcNote}Research use only.</p>`;
    html += `<div class="downloads" style="margin-top:10px"><a href="${f("morphometry_branches.csv")}" download>All branches (CSV)</a>
      <a href="${f("morphometry.json")}" download>Summary (JSON)</a></div>`;
    $("morph-body").innerHTML = html;

    // click a branch row: move the crosshair to that branch (needs the viewer)
    const nv = window.airwayViewer;
    if (nv && rows.length) {
      const trs = $("morph-body").querySelectorAll(".mwrap tbody tr");
      trs.forEach((tr, i) => {
        tr.classList.add("link");
        tr.onclick = () => jumpToBranch(nv, rows[i].branch_instance_id);
      });
    }
  }

  async function jumpToBranch(nv, branchId) {
    // Centroid of the branch in world mm (voxel index through the header
    // affine, so the file's own orientation does not matter), then crosshair.
    const src = window._branchesImg;
    if (!src || !src.img || !src.hdr) return;
    const d = src.hdr.dims, A = src.hdr.affine, data = src.img;
    let n = 0, sx = 0, sy = 0, sz = 0;
    for (let i = 0; i < data.length; i++) {
      if (data[i] === branchId) {
        sx += i % d[1]; sy += Math.floor(i / d[1]) % d[2]; sz += Math.floor(i / (d[1] * d[2])); n++;
      }
    }
    if (!n) return;
    const v = [sx / n, sy / n, sz / n];
    const mm = [0, 1, 2].map(r => A[r][0] * v[0] + A[r][1] * v[1] + A[r][2] * v[2] + A[r][3]);
    nv.scene.crosshairPos = nv.mm2frac(mm);
    nv.drawScene();
    const vox = nv.frac2vox(nv.scene.crosshairPos);
    $("readout").textContent = `voxel ${Array.from(vox).map(Math.round).join(", ")} · ` +
      `${mm.map(x => x.toFixed(1)).join(", ")} mm · branch #${branchId}`;
    $("gl").scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // ------------------------------------------------------------- history
  async function history() {
    let rows = [];
    try { rows = await (await fetch("/api/jobs?limit=200", { cache: "no-store" })).json(); }
    catch (_) { $("hmsg").textContent = "Could not load the list."; return; }
    if (!rows.length) { $("hmsg").textContent = "No uploads yet."; return; }
    $("hmsg").textContent = `${rows.length} upload${rows.length === 1 ? "" : "s"}. Results stay available while they are kept on the server.`;
    const when = (iso) => iso ? new Date(iso).toLocaleString() : "";
    $("hist").innerHTML = table(["Uploaded", "File", "Age", "State", "Progress", ""],
      rows.map(r => [when(r.created), esc(r.filename || ""), r.age_years != null ? `${r.age_years} y` : "",
                     esc(r.state || ""), r.state === "done" ? "100%" : `${Math.round((r.progress || 0) * 100)}%`,
                     `<a href="/jobs/${r.id}">open</a>`]));
  }

  return { upload, job, history };
})();

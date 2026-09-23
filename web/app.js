/* Two pages, one script. No framework: the app is an upload form and a poller. */
const AirwayApp = (() => {
  const $ = (id) => document.getElementById(id);
  const hex = (h) => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

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
    let shown = false, morphShown = false;
    const fmt = (s) => s < 60 ? `${Math.round(s)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;

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
      if (s.results_ready && !shown) { shown = true; showResults(id, s.results || {}); }
      const mst = (s.stages || []).find(x => x.name === "morphometry");
      if (shown && mst && !morphShown) {
        $("morph").hidden = false;
        if (mst.state === "done") { morphShown = true; showMorphometry(id); }
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
  async function showResults(id, r) {
    $("results").hidden = false;
    const f = (n) => `/api/jobs/${id}/files/${n}`;
    const stats = [];
    if (r.airway_volume_ml != null) stats.push([`${r.airway_volume_ml} mL`, "airway lumen volume"]);
    if (r.branches != null) stats.push([r.branches, "branches named"]);
    if (r.lobes) stats.push([r.lobes.filter(l => l.name !== "trachea").length, "lobes reached"]);
    if (r.view_spacing_mm) stats.push([`${r.view_spacing_mm} mm`, "display resolution"]);
    if (r.source_orientation) stats.push([r.source_orientation, "output orientation"]);
    $("stats").innerHTML = stats.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
    const dl = [
      ["airway_results.zip", "Everything (zip)", "zip"],
      ["airway_mask.nii.gz", "Airway mask"], ["airway_lobar.nii.gz", "Lobar labels"],
      ["airway_segmental.nii.gz", "Segmental labels"], ["airway_branches.nii.gz", "Branch instances"],
      ["airway_centreline.nii.gz", "Centreline"], ["mesh_airway.mz3", "Airway surface (MZ3)"],
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
    const boot = () => viewer(id, f, { lobNames, lobColours, segNames, segColours, brColours, brNames,
                                        airwayColour, centreColour: pal.centreline || "#f5e663",
                                        meshes: r.meshes || [] });
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

    // ---- state ---------------------------------------------------------------
    const OVERLAYS = {
      airway:    { file: "airway_mask_view.nii.gz",      colours: { 1: L.airwayColour }, names: { 1: "airway" } },
      lobar:     { file: "airway_lobar_view.nii.gz",     colours: L.lobColours, names: L.lobNames },
      segmental: { file: "airway_segmental_view.nii.gz", colours: L.segColours, names: L.segNames },
      branches:  { file: "airway_branches_view.nii.gz",  colours: L.brColours, names: L.brNames },
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
      const img = await NVImage.loadFromUrl({ url: f(o.file), name: o.file, colormap: "gray", opacity: 1 });
      img.setColormapLabel(labelColormap(o.colours));
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
        if (ct) nv.setOpacity(0, planes ? 1 : 0);
        nv.opts.renderOverlayBlend = planes ? 1 : 0;
      }
      if (overlay && nv.volumes.includes(overlay)) nv.setOpacity(nv.volumes.indexOf(overlay), opacity());
      nv.opts.show3Dcrosshair = planes && $("ck-xh").checked;
      nv.drawScene();
    };
    const showMeshes = () => {
      const on = $("ck-mesh").checked, lob = overlayKey === "lobar", op = $("mesh-op").value / 100;
      for (const m of nv.meshes) {
        const base = meshBase(m);
        const vis = on && (base === "airway" ? !lob : (lob && !!lobeOn[+base.split("_")[1]]));
        if (m.visible !== vis) nv.setMeshProperty(m.id, "visible", vis);
        if (m.opacity !== op) nv.setMeshProperty(m.id, "opacity", op);
      }
      nv.drawScene();
    };
    const legend = (key) => {
      const o = OVERLAYS[key];
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
      } catch (e) { fail(`${key} overlay`, e); }
    };
    const setLayout = (v) => {
      layout = v;
      if (v === "grid") { nv.opts.multiplanarShowRender = 1; nv.setSliceType(3); }
      else nv.setSliceType(Number(v));
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
    let branchesImg = null;
    nv.onLocationChange = (d) => {
      try {
        const v = d.vox, mm = d.mm;
        const hu = d.values && d.values[0] ? Math.round(d.values[0].value) : "";
        let br = "";
        if (branchesImg && v) {
          const val = branchesImg.getValue(Math.round(v[0]), Math.round(v[1]), Math.round(v[2]));
          if (val > 0) br = ` · ${L.brNames[String(val)] || `branch ${val}`} (#${val})`;
        }
        say(`voxel ${Array.from(v).map(Math.round).join(", ")} · ` +
            `${Array.from(mm).slice(0, 3).map(x => x.toFixed(1)).join(", ")} mm · ${hu} HU${br}`);
      } catch (_) {}
    };

    // ---- load, in order of what the user wants to see first --------------------------
    try {
      say("loading the CT…  (large scans take a moment)");
      await nv.loadVolumes([{ url: f("image_view.nii.gz"), colormap: "gray", opacity: 1 }]);
      ct = nv.volumes[0];
      setWindow(-1000, 300);
    } catch (e) { fail("CT", e); return; }
    await applyVisibility();                 // slice planes in the grid's 3D pane
    await setOverlay("airway");
    try {
      if (overlayKey) say("loading the airway surface…");
      const meshList = [{ url: f("mesh_airway.mz3"), name: meshName("airway"), rgba255: [...hex(L.airwayColour), 255], visible: true }];
      for (const m of L.meshes.filter(m => m.kind === "lobe")) {
        meshList.push({ url: f(m.file), name: meshName(`lobe_${m.label}`),
                        rgba255: [...hex(L.lobColours[String(m.label)] || "#999999"), 255], visible: false });
      }
      await nv.loadMeshes(meshList);
      showMeshes();
      if (overlayKey) legend(overlayKey);
      say("Click a slice to read position and value. Left-drag rotates the 3D view.");
    } catch (e) { fail("surfaces", e); }
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
    $("morph-msg").textContent = "";

    const stats = [
      [m.branch_count, "branches"],
      [`${num(m.total_centreline_length_mm, 0)} mm`, "total centreline length"],
      [m.max_generation, "deepest generation"],
      [m.airway_lumen_volume_ml != null ? `${num(m.airway_lumen_volume_ml)} mL` : "–", "lumen volume"],
      [xok ? `${num(x.trachea_length_intrathoracic_mm)} mm` : ok ? `${num(c.trachea.length_mm)} mm` : "–",
       xok ? "intrathoracic tracheal length" : "tracheal length"],
      [xok ? `${num(x.trachea_mean_diameter_mm)} mm` : ok ? `${num(c.trachea.diameter_edt_mean_mm)} mm` : "–",
       xok ? "tracheal diameter (cross section)" : "tracheal diameter (inscribed)"],
      [xok && x.airway_to_lung_volume_percent != null ? `${num(x.airway_to_lung_volume_percent, 2)} %` : "–", "airway / lung volume"],
      [m.skeleton_fractal_dimension != null ? num(m.skeleton_fractal_dimension, 2) : "–", "skeleton fractal dimension"],
    ];
    let html = `<div class="stats">${stats.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("")}</div>`;
    html += `<div class="mgrid">`;

    // central airways
    if (ok) {
      const row = (name, b, dx, lx) => [name, num(lx != null ? lx : b.length_mm), num(b.diameter_edt_mean_mm),
        dx != null ? num(dx) : "–", b.angle_deg != null ? num(b.angle_deg, 0) : "–"];
      let rows = [row("Trachea", c.trachea, x.trachea_mean_diameter_mm, x.trachea_length_intrathoracic_mm),
                  row("Right main bronchus", c.rmb, x.rmb_mean_diameter_mm, x.rmb_length_mm),
                  row("Left main bronchus", c.lmb, x.lmb_mean_diameter_mm, x.lmb_length_mm)];
      html += `<div><h3>Central airways</h3>${table(["", "Length mm", "Ø inscribed mm", "Ø cross section mm", "Angle °"], rows)}`;
      if (xok) {
        const ratios = [
          ["Tracheal short / long axis", `${num(x.trachea_short_axis_mm)} / ${num(x.trachea_long_axis_mm)} mm`],
          ["Tracheal eccentricity (short / long)", num(x.trachea_eccentricity, 2)],
          ["Tracheal taper (distal / proximal Ø)", num(x.trachea_taper_distal_over_proximal, 2)],
          ["Tracheal length / diameter", num(x.trachea_length_over_diameter, 1)],
          ["Daughter / parent Ø at carina", num(x.daughter_over_parent_diameter, 2)],
          ["Right-left asymmetry (R−L)/(R+L)", num(x.right_left_asymmetry, 2)],
          ["Interbronchial angle (proxy, sum of local angles)", c.interbronchial_angle_proxy_deg != null ? `${num(c.interbronchial_angle_proxy_deg, 0)} °` : "–"],
          ["Segmented lung volume", `${num(x.lung_volume_ml, 0)} mL`],
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
      (m.per_generation || []).map(g => [g.generation, g.n, num(g.diameter_mean_mm), num(g.length_mean_mm),
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
      html += `<h3 style="font-size:13px;margin:18px 0 6px">All branches <span class="hint">(click a row to jump there)</span></h3><div class="mwrap">` +
        table(["#", "Gen", "Name", "Lobe", "Segment", "Length mm", "Ø mean mm", "Ø min mm", "Tortuosity", "Angle °", "Volume mm³", "QC"],
          rows.map(r => [r.branch_instance_id, r.generation, esc(r.branch_name), esc(r.lobar_name), esc(r.segmental_name),
            num(r.centerline_length_mm), num(r.diameter_mean_mm), num(r.diameter_min_mm), num(r.tortuosity_length_ratio, 2),
            r.branch_angle_deg != null ? num(r.branch_angle_deg, 0) : "–", num(r.volume_mm3, 0), esc(r.geometry_qc_flags || "")])) + `</div>`;
    }
    html += `<p class="mnote">Measured on the segmented lumen with the centreline of the fused mask. Diameters:
      "inscribed" is twice the distance-transform radius along the centreline, "cross section" is the mean of short and
      long axis of the lumen in planes perpendicular to the centreline. Angles are between a branch and its parent.
      Research use only.</p>`;
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

  return { upload, job };
})();

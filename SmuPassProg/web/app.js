(function () {
  const $ = (id) => document.getElementById(id);

  const els = {
    btnRunAll: $("btnRunAll"),
    statusText: $("statusText"),
    kpiRow: $("kpiRow"),
    vTotal: $("vTotal"),
    vEligible: $("vEligible"),
    vAccept: $("vAccept"),
    vCut: $("vCut"),
    imgScatter: $("imgScatter"),
    imgHeat: $("imgHeat"),
    logOutput: $("logOutput"),
    tblBody: $("tblBody"),
    trimHint: $("trimHint"),

    singleBox: $("singleBox"),
    sTotal: $("sTotal"),
    sPassTotal: $("sPassTotal"),
    sFinal: $("sFinal"),
    sDetail: $("sDetail"),
  };

  function setStatus(msg, kind) {
    els.statusText.classList.remove("ok", "bad");
    if (kind === "ok") els.statusText.classList.add("ok");
    if (kind === "bad") els.statusText.classList.add("bad");
    els.statusText.textContent = msg || "";
  }

  async function fetchJson(url, options) {
    const res = await fetch(url, options);
    const text = await res.text();
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}\n${text.slice(0, 800)}`);
    }
    try {
      return JSON.parse(text);
    } catch (e) {
      throw new Error(`JSON parse failed\n${text.slice(0, 800)}`);
    }
  }

  function fmtNum(x, digits = 3) {
    if (typeof x !== "number") return x ?? "-";
    return x.toFixed(digits);
  }

  function statusPill(status) {
    const s = String(status || "");
    let cls = "";
    if (s === "PASS") cls = "pass";
    else if (s === "FAIL") cls = "fail";
    else if (s.startsWith("TRIMMED")) cls = "trim";
    return `<span class="pill ${cls}">${escapeHtml(s)}</span>`;
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function renderCohort(cohort) {
    if (!cohort) return;

    const summary = cohort.summary || {};
    const trim = summary.trim || cohort.trim || {};
    const results = cohort.results || [];

    els.kpiRow.style.display = "grid";
    els.vTotal.textContent = summary.total_files ?? cohort.total_files ?? "-";
    els.vEligible.textContent = trim.eligible_count ?? "-";
    els.vAccept.textContent = summary.accepted_count ?? cohort.accepted_count ?? "-";
    els.vCut.textContent = summary.cutoff_total_score ?? cohort.cutoff_total_score ?? "-";

    // trimming hint
    const mode = trim.mode ?? "-";
    const pct = trim.pct ?? trim.trim_pct ?? "-";
    const top = trim.top_trimmed ?? trim.top_trimmed_count ?? trim.top ?? "-";
    const bottom = trim.bottom_trimmed ?? trim.bottom_trimmed_count ?? trim.bottom ?? "-";
    els.trimHint.textContent = `절사 모드=${mode}, 절사비율=${pct}, 상위 제외=${top}, 하위 제외=${bottom}, 평가대상=${trim.eligible_count ?? "-"}`;

    // table
    if (!Array.isArray(results) || results.length === 0) {
      els.tblBody.innerHTML = `<tr><td colspan="7" style="color:var(--muted)">results가 비어 있습니다.</td></tr>`;
      return;
    }

    els.tblBody.innerHTML = results.map((r) => {
      const rank = r.rank ?? "";
      const fn = r.filename ?? "";
      const total = (typeof r.total_score === "number") ? r.total_score.toFixed(3) : (r.total_score ?? "");
      const status = r.status ?? (r.pass ? "PASS" : "FAIL");
      const pc1 = (typeof r.pc1 === "number") ? r.pc1.toFixed(3) : (r.pc1 ?? "");
      const pc2 = (typeof r.pc2 === "number") ? r.pc2.toFixed(3) : (r.pc2 ?? "");
      const stype = r.student_type ?? "";
      return `
        <tr>
          <td>${rank}</td>
          <td>${escapeHtml(fn)}</td>
          <td>${escapeHtml(total)}</td>
          <td>${statusPill(status)}</td>
          <td>${escapeHtml(pc1)}</td>
          <td>${escapeHtml(pc2)}</td>
          <td>${escapeHtml(stype)}</td>
        </tr>
      `;
    }).join("");
  }

  function renderInput(input) {
    if (!input) return;

    els.singleBox.style.display = "block";

    const asNum = (v) => {
      if (typeof v === "number" && Number.isFinite(v)) return v;
      if (typeof v === "string") {
        const n = Number(v);
        if (Number.isFinite(n)) return n;
      }
      return undefined;
    };

    // input_result.json 형식은 프로젝트에 따라 다를 수 있어서 최대한 유연하게 표시
    const totalRaw = input.total_score ?? input.final_score ?? input.total ?? input?.result?.total_score;
    const cutoffRaw = input.cutoff_total_score ?? input?.decision?.cutoff_total_score ?? input?.cutoff?.total_score;
    const total = asNum(totalRaw) ?? totalRaw;
    const cutoff = asNum(cutoffRaw) ?? cutoffRaw;

    // 총점 통과 여부(명시돼 있으면 우선, 없으면 계산)
    const passTotalExplicit = input.pass_total ?? input?.decision?.pass_total ?? input?.result?.pass_total;
    const passTotal = (typeof passTotalExplicit === "boolean")
      ? passTotalExplicit
      : (typeof total === "number" && typeof cutoff === "number" ? total >= cutoff : undefined);

    // PCA
    const pc1 = asNum(input.pc1 ?? input?.pca?.pc1 ?? input?.result?.pc1) ?? (input.pc1 ?? input?.pca?.pc1 ?? input?.result?.pc1);
    const pc2 = asNum(input.pc2 ?? input?.pca?.pc2 ?? input?.result?.pc2) ?? (input.pc2 ?? input?.pca?.pc2 ?? input?.result?.pc2);
    const pc1Cut = asNum(input.pc1_cutoff ?? input?.pca?.pc1_cutoff ?? input?.decision?.pc1_cutoff) ?? (input.pc1_cutoff ?? input?.pca?.pc1_cutoff ?? input?.decision?.pc1_cutoff);

    // PC1 통과 여부(명시돼 있으면 우선, 없으면 계산)
    const passPc1Explicit = input.pca_pass_pc1 ?? input?.pca?.pass_pc1 ?? input?.result?.pca_pass_pc1;
    const passPc1 = (typeof passPc1Explicit === "boolean")
      ? passPc1Explicit
      : (typeof pc1 === "number" && typeof pc1Cut === "number" ? pc1 >= pc1Cut : undefined);

    // 최종 합격(명시돼 있으면 우선, 없으면 계산)
    const finalPassExplicit = input.final_pass ?? input?.decision?.final_pass ?? input?.result?.final_pass;
    const finalPass = (typeof finalPassExplicit === "boolean")
      ? finalPassExplicit
      : (passTotal === undefined ? undefined : Boolean(passTotal && (passPc1 ?? true)));

    els.sTotal.textContent = (typeof total === "number") ? total.toFixed(3) : String(total ?? "-");
    els.sPassTotal.textContent = String(passTotal ?? "-");
    els.sFinal.textContent = String(finalPass ?? "-");

    const lines = [];
    if (input.file || input.filename) lines.push(`file: ${input.file || input.filename}`);
    if (typeof cutoff !== "undefined") lines.push(`총점 컷: ${cutoff}`);
    if (typeof pc1Cut !== "undefined") lines.push(`PC1 컷: ${pc1Cut}`);
    // 현재 알고리즘은 PC2로 컷을 두지 않습니다. (PC2는 분포/유형 파악용)
    lines.push("PC2 컷: 없음 (PC2는 유형/분포용)");
    if (typeof pc1 !== "undefined") lines.push(`PC1: ${pc1}`);
    if (typeof pc2 !== "undefined") lines.push(`PC2: ${pc2}`);
    if (typeof passPc1 !== "undefined") lines.push(`PC1 컷 통과: ${passPc1}`);
    if (typeof finalPass !== "undefined") lines.push(`최종 합격: ${finalPass}`);
    els.sDetail.textContent = lines.join("\n");
  }

  function renderImages(artifacts) {
    const ts = Date.now();
    const scatter = artifacts?.pca_scatter_url || "/output/pca_scatter.png";
    const heat = artifacts?.pca_heatmap_url || "/output/pca_heatmap.png";
    els.imgScatter.src = `${scatter}?t=${ts}`;
    els.imgHeat.src = `${heat}?t=${ts}`;
  }

  els.btnRunAll.addEventListener("click", async (e) => {
    e.preventDefault();
    setStatus("분석 실행 중… (main.py가 파일 선택 창을 띄우면 1회만 선택하세요)", null);
    els.btnRunAll.disabled = true;
    els.logOutput.textContent = "실행 중입니다…";

    try {
      const data = await fetchJson("/api/run_all", { method: "POST" });

      // stdout 그대로 보여주기
      els.logOutput.textContent = data.stdout || "(stdout 비어 있음)";

      renderImages(data.artifacts);
      renderCohort(data.cohort);
      renderInput(data.input);

      setStatus("완료", "ok");
    } catch (err) {
      els.logOutput.textContent = err.message || String(err);
      setStatus(`실패:\n${err.message}`, "bad");
    } finally {
      els.btnRunAll.disabled = false;
    }
  });
})();

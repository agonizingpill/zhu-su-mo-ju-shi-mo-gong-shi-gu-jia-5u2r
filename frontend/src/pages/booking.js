import {send} from '../api.js';
import {esc, number, money} from '../utils/format.js';
import {notify, empty} from '../components/common.js';
import {SAMPLE_ORDER_STATES} from '../constants.js';

const pct = x => x == null ? '—' : (Number(x) * 100).toFixed(2) + '%';

export function render(s) {
  const o = s.sampleOrders.find(x => x.id === s.bookingSelected) || s.sampleOrders[0];
  const bookable = s.estimates.filter(e => e.status === 'committed' &&
    !s.executions.some(x => x.estimate_id === e.id && x.status !== 'running'));
  return `<div class="pagehead"><div class="kicker">10 / PRE-COMPLETION SAMPLE BOOKING</div><h1>完工前订样与分批发样</h1><p>按模具已签实绩训练良品率下界，计划射次换算可承诺件数；样本不足先挂起，完工按承诺先后分货，签结后分批发样、重试不重复扣件。</p></div>
<div class="booking-grid">
  <section class="section">
    <h2>良品率下界（按模具）</h2>
    <form id="yield-train">
      <label>模具<select name="mould_id">${s.moulds.map(m => `<option value="${m.id}">${esc(m.code)} · ${esc(m.name)}</option>`).join('')}</select></label>
      <label>单侧分位 z（1.645≈95%）<input name="z" type="number" step="0.001" min="0.001" max="4" value="1.645" required></label>
      <button ${s.user.role === 'viewer' ? 'disabled' : ''}>训练/重训下界</button>
    </form>
    <div class="rule">仅已签结实绩入训；同模具留一验证，少于 5 条样本拒绝训练，相关订样单保持挂起。</div>
    <div class="execution-list">${s.yieldModels.map(m => `<button class="execution-row" data-yield="${m.id}">
      <strong>${esc(m.mould_code)} · v${m.version}</strong>
      <span>下界 <b class="mono">${pct(m.lower_rate)}</b> · 均值 ${pct(m.mean_rate)} · σ ${number(m.std_rate)}</span>
      <small>样本 ${m.sample_size} 条 · LOO 覆盖 ${pct(m.metrics.loo_coverage)}（${m.metrics.loo_covered}/${m.metrics.loo_total}）</small>
    </button>`).join('') || empty('先选择模具训练良品率下界。')}</div>
  </section>

  <section class="section">
    <h2>订样台账</h2>
    <form id="order-create">
      <label>已承诺且未完工估单<select name="estimate_id">${bookable.map(e => `<option value="${e.id}">${esc(e.code)} · ${esc((s.moulds.find(m => m.id === e.mould_id) || {}).code || '')}</option>`).join('')}</select></label>
      <div class="fields">
        <label>订样单号<input name="order_no" placeholder="SO-261103" required></label>
        <label>客户<input name="customer" placeholder="华远质量部" required></label>
      </div>
      <div class="fields">
        <label>计划射次<input name="planned_shots" type="number" min="1" value="100" required></label>
        <label>需求件数<input name="requested_qty" type="number" min="1" value="20" required></label>
      </div>
      <label>备注<input name="note" placeholder="可选"></label>
      <button class="primary" ${s.user.role === 'viewer' || !bookable.length ? 'disabled' : ''}>提交订样</button>
    </form>
    <div class="rule">可承诺 = min(需求, floor(计划射次 × 下界))；同试模并发订样累计不得超占，完工后名单锁定。</div>
    <div class="execution-list">${s.sampleOrders.map(x => `<button class="execution-row ${x.id === (o && o.id) ? 'selected' : ''}" data-order="${x.id}">
      <strong>${esc(x.order_no)} · ${esc(x.customer)}</strong>
      <span class="status ${x.status}">${SAMPLE_ORDER_STATES[x.status] || x.status}</span>
      <small>${esc(x.estimate_code)} · 承诺 ${x.promised_qty} / 待发 ${x.reserved_qty} / 已发 ${x.dispatched_qty}${x.shortfall_qty ? ` · <b class="gap">缺口 ${x.shortfall_qty}</b>` : ''}</small>
    </button>`).join('') || empty('暂无订样单。')}</div>
    <button id="re-evaluate" class="small" style="margin-top:14px" ${s.user.role === 'viewer' ? 'disabled' : ''}>重评挂起订样单</button>
  </section>

  <section class="section">
    ${o ? renderDetail(s, o) : empty('左侧选择订样单查看承诺、缺口与发样批次。')}
  </section>
</div>
<div class="section" style="margin-top:28px">
  <h2>缺口看板</h2>
  <div class="booking-gaps">
    <div><h3>完工锁定后的分货缺口（较早承诺优先分货）</h3>${s.sampleShortfalls.shortfall_orders.length ? `<table><thead><tr><th>单号</th><th>估单</th><th>承诺</th><th>已分</th><th>缺口</th><th>状态</th></tr></thead><tbody>${s.sampleShortfalls.shortfall_orders.map(x => `<tr><td>${esc(x.order_no)}</td><td>${esc(x.estimate_code)}</td><td>${x.promised_qty}</td><td>${x.reserved_qty + x.dispatched_qty}</td><td class="gap">${x.shortfall_qty}</td><td>${SAMPLE_ORDER_STATES[x.status]}</td></tr>`).join('')}</tbody></table>` : empty('当前无分货缺口。')}</div>
    <div><h3>样本不足挂起单（训练达标后可重评）</h3>${s.sampleShortfalls.suspended_orders.length ? `<table><thead><tr><th>单号</th><th>估单</th><th>需求</th><th>客户</th></tr></thead><tbody>${s.sampleShortfalls.suspended_orders.map(x => `<tr><td>${esc(x.order_no)}</td><td>${esc(x.estimate_code)}</td><td>${x.requested_qty}</td><td>${esc(x.customer)}</td></tr>`).join('')}</tbody></table>` : empty('没有挂起中的订样单。')}</div>
  </div>
</div>`;
}

function renderDetail(s, o) {
  const canDispatch = s.user.role !== 'viewer' && ['locked', 'locked_short', 'completed'].includes(o.status) && o.reserved_qty > 0;
  return `<h2>${esc(o.order_no)} · 订样详情</h2>
  <div class="ribbon">
    <div>状态<strong>${SAMPLE_ORDER_STATES[o.status] || o.status}</strong></div>
    <div>承诺件数<strong class="mono">${o.promised_qty}</strong></div>
    <div>待发预留<strong class="mono">${o.reserved_qty}</strong></div>
    <div>已发件数<strong class="mono">${o.dispatched_qty}</strong></div>
    <div>缺口<strong class="mono gap">${o.shortfall_qty}</strong></div>
  </div>
  <p class="muted">${esc(o.estimate_code)} · ${esc(o.mould_code)} · 计划射次 ${o.planned_shots} · 下界快照 ${pct(o.rate_snapshot)}（模型 v${o.model_version || '—'}）· 预期良品 ${o.expected_good_qty}</p>
  <form id="dispatch-create" data-id="${o.id}">
    <div class="fields">
      <label>批次号（重试同号不重复扣件）<input name="batch_no" placeholder="BATCH-001" required></label>
      <label>本批件数<input name="qty" type="number" min="1" max="${Math.max(1, o.reserved_qty)}" value="${Math.max(1, o.reserved_qty)}" required></label>
    </div>
    <button class="primary" ${canDispatch ? '' : 'disabled'}>${o.status === 'committed' || o.status === 'pending' ? '完工签结后方可发样' : '分批发样'}</button>
  </form>
  <h3 style="margin-top:22px">发样批次（旧交付已发件数照常计入良品池）</h3>
  ${o.batches.length ? `<table><thead><tr><th>批次</th><th>件数</th><th>状态</th><th>签收人/交接号</th><th>操作</th></tr></thead><tbody>${o.batches.map(b => `<tr>
    <td>${esc(b.batch_no)}</td><td>${b.qty}</td><td>${b.state === 'received' ? '已签收' : '已发样待签收'}</td>
    <td>${b.received_by ? esc(b.received_by + ' / ' + b.tracking) : '—'}</td>
    <td>${b.state === 'dispatched' && s.user.role !== 'viewer' ? `<form class="inline-form" data-receive="${b.id}"><input name="tracking" placeholder="交接号" required><input name="received_by" placeholder="签收人" required><button class="small">登记签收</button></form>` : '—'}</td>
  </tr>`).join('')}</tbody></table>` : empty('尚无发样批次。')}
  <p class="rule">发样在结算签结后开放；同批次号重复提交直接返回原批次，不二次扣减预留。签收照常登记，重复签收会被拒绝。</p>`;
}

export function bind(s, draw) {
  const act = async (fn, msg) => {
    try { await fn(); await draw(); notify(msg); } catch (e) { notify(e.message); }
  };
  document.querySelectorAll('[data-order]').forEach(b => b.onclick = () => { s.bookingSelected = +b.dataset.order; draw(); });

  const trainForm = document.querySelector('#yield-train');
  if (trainForm) trainForm.onsubmit = e => {
    e.preventDefault();
    const d = Object.fromEntries(new FormData(e.target));
    d.mould_id = +d.mould_id; d.z = +d.z;
    act(async () => {
      const r = await send('/yield-models/train', d);
      if (r.failed.length) notify('部分模具未训练：' + r.failed.map(f => f.reason).join('；'));
    }, '良品率下界训练完成，挂起单可重评');
  };

  const createForm = document.querySelector('#order-create');
  if (createForm) createForm.onsubmit = e => {
    e.preventDefault();
    const d = Object.fromEntries(new FormData(e.target));
    d.estimate_id = +d.estimate_id; d.planned_shots = +d.planned_shots; d.requested_qty = +d.requested_qty;
    act(async () => {
      const x = await send('/sample-orders', d);
      s.bookingSelected = x.id;
      notify(x.status === 'pending' ? '样本不足，订样单已挂起' : `已按下界承诺 ${x.promised_qty} 件`);
    }, '');
  };

  const reeval = document.querySelector('#re-evaluate');
  if (reeval) reeval.onclick = () => act(() => send('/sample-orders/evaluate', {}), '挂起订样单已重评');

  const dispatchForm = document.querySelector('#dispatch-create');
  if (dispatchForm) dispatchForm.onsubmit = e => {
    e.preventDefault();
    const d = Object.fromEntries(new FormData(e.target));
    d.qty = +d.qty;
    act(async () => {
      const r = await send('/sample-orders/' + dispatchForm.dataset.id + '/dispatches', d);
      notify(r.idempotent_replay ? '同批次重试，未重复扣件' : `发样 ${r.batch.qty} 件，预留已扣减`);
    }, '');
  };

  document.querySelectorAll('form[data-receive]').forEach(f => f.onsubmit = e => {
    e.preventDefault();
    const d = Object.fromEntries(new FormData(f));
    act(() => send('/sample-dispatches/' + f.dataset.receive + '/receive', d), '客户签收已登记');
  });
}

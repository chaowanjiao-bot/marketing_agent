from __future__ import annotations

from collections import Counter
from html import escape
from typing import Any

from .task_store import TaskStore


STATUS_COLORS = {
    "completed": "#25c281", "waiting_for_review": "#f6b94a", "running": "#5b8cff",
    "queued": "#8b9bb4", "failed": "#f0636a", "aborted": "#ef8b55",
}


class DashboardService:
    def __init__(self, store: TaskStore) -> None:
        self.store = store

    def task_snapshot(self, task_id: str) -> dict[str, Any]:
        status = self.store.status(task_id)
        result = self.store.result(task_id)
        request = self.store.request(task_id)
        return {
            "task_id": task_id,
            "status": status.get("status", "unknown"),
            "phase": status.get("phase", ""),
            "prompt": request.prompt,
            "candidate_count": request.candidate_count,
            "output_formats": [item.value for item in request.output_formats],
            "review_required": request.review_required,
            "review_round": request.review_round,
            "best_score": result.get("best_score") if result else None,
            "best_asset_id": result.get("best_asset_id") if result else None,
            "terminal_reason": result.get("terminal_reason") if result else None,
            "review_status": result.get("review_status") if result else None,
        }

    def summary(self) -> dict[str, Any]:
        snapshots = [self.task_snapshot(task_id) for task_id in self.store.task_ids()]
        counts = Counter(snapshot["status"] for snapshot in snapshots)
        scores = [float(snapshot["best_score"]) for snapshot in snapshots
                  if snapshot["best_score"] is not None]
        return {
            "total_tasks": len(snapshots),
            "status_counts": dict(counts),
            "average_best_score": round(sum(scores) / len(scores), 4) if scores else None,
            "review_queue": counts.get("waiting_for_review", 0),
            "recent_tasks": snapshots[:50],
        }

    def render_index(self) -> str:
        summary = self.summary()
        rows = "".join(self._task_row(item) for item in summary["recent_tasks"])
        avg = summary["average_best_score"]
        cards = [
            ("任务总数", summary["total_tasks"]),
            ("平均最佳分", f"{avg:.3f}" if avg is not None else "—"),
            ("待人工审核", summary["review_queue"]),
            ("已完成", summary["status_counts"].get("completed", 0)),
        ]
        card_html = "".join(
            f'<section class="metric"><span>{escape(label)}</span><strong>{value}</strong></section>'
            for label, value in cards
        )
        empty = '<tr><td colspan="6" class="empty">暂无任务</td></tr>' if not rows else ""
        return self._page("营销创意 Agent 看板", f"""
          <header><div><p class="eyebrow">MARKETING CREATIVE AGENT</p>
          <h1>实验与决策看板</h1><p>候选、画幅、评分与人工审核状态</p></div>
          <a class="api" href="/dashboard/api/summary">JSON API</a></header>
          <div class="metrics">{card_html}</div>
          <section class="panel"><div class="panel-title"><h2>最近任务</h2>
          <span>{summary['total_tasks']} total</span></div>
          <div class="table-wrap"><table><thead><tr><th>任务</th><th>状态</th>
          <th>最佳分</th><th>候选</th><th>画幅</th><th>审核</th></tr></thead>
          <tbody>{rows}{empty}</tbody></table></div></section>""")

    def render_task(self, task_id: str) -> str:
        snapshot = self.task_snapshot(task_id)
        result = self.store.result(task_id) or {}
        assets = {item["asset_id"]: item for item in result.get("assets", [])}
        format_cards = "".join(
            self._format_card(task_id, item, assets) for item in result.get("format_summaries", [])
        ) or '<p class="empty">结果尚未生成</p>'
        candidate_rows = "".join(self._candidate_row(item) for item in result.get(
            "candidate_summaries", []
        )) or '<tr><td colspan="6" class="empty">单候选任务</td></tr>'
        trace_rows = "".join(
            f'<li><code>{escape(str(item.get("event", "event")))}</code>'
            f'<span>{escape(self._trace_detail(item))}</span></li>'
            for item in result.get("trace", [])[-30:]
        ) or '<li class="empty">暂无轨迹</li>'
        return self._page(f"任务 {task_id}", f"""
          <nav><a href="/dashboard">← 返回总览</a></nav>
          <header><div><p class="eyebrow">{escape(task_id)}</p><h1>{escape(snapshot['prompt'])}</h1>
          <p>{escape(str(snapshot['terminal_reason'] or '执行中'))}</p></div>
          {self._badge(str(snapshot['status']))}</header>
          <div class="metrics">
            <section class="metric"><span>最佳分</span><strong>{self._score(snapshot['best_score'])}</strong></section>
            <section class="metric"><span>候选数</span><strong>{snapshot['candidate_count']}</strong></section>
            <section class="metric"><span>审核轮次</span><strong>{snapshot['review_round']}</strong></section>
            <section class="metric"><span>画幅数</span><strong>{len(snapshot['output_formats'])}</strong></section>
          </div>
          <section class="panel"><div class="panel-title"><h2>各画幅最佳结果</h2></div>
          <div class="formats">{format_cards}</div></section>
          <section class="panel"><div class="panel-title"><h2>候选比较</h2></div>
          <div class="table-wrap"><table><thead><tr><th>画幅</th><th>候选</th><th>Seed</th>
          <th>评分</th><th>文字合规</th><th>选择</th></tr></thead><tbody>{candidate_rows}</tbody></table></div></section>
          <section class="panel"><div class="panel-title"><h2>Agent 决策轨迹</h2></div>
          <ol class="trace">{trace_rows}</ol></section>""")

    def _task_row(self, item: dict[str, Any]) -> str:
        return (f'<tr><td><a href="/dashboard/tasks/{escape(item["task_id"])}">'
                f'{escape(item["prompt"][:48])}</a><small>{escape(item["task_id"])}</small></td>'
                f'<td>{self._badge(item["status"])}</td><td>{self._score(item["best_score"])}</td>'
                f'<td>{item["candidate_count"]}</td><td>{escape(", ".join(item["output_formats"]))}</td>'
                f'<td>{escape(str(item["review_status"] or "—"))}</td></tr>')

    def _format_card(self, task_id: str, item: dict[str, Any], assets: dict[str, Any]) -> str:
        asset_id = item.get("best_asset_id")
        asset = assets.get(asset_id, {})
        image = (f'<img src="/tasks/{escape(task_id)}/assets/{escape(str(asset_id))}" '
                 f'alt="{escape(str(item.get("output_format")))} result">') if asset_id else ""
        return (f'<article class="format-card">{image}<div><strong>{escape(str(item.get("output_format")))}</strong>'
                f'<span>{item.get("width")} × {item.get("height")}</span>'
                f'<span>score {self._score(item.get("best_score"))}</span></div></article>')

    def _candidate_row(self, item: dict[str, Any]) -> str:
        return (f'<tr><td>{escape(str(item.get("output_format")))}</td>'
                f'<td>#{int(item.get("candidate_index", 0)) + 1}</td><td>{item.get("seed")}</td>'
                f'<td>{self._score(item.get("best_score"))}</td>'
                f'<td>{"✓" if item.get("compliant") else "✕"}</td>'
                f'<td>{"BEST" if item.get("selected") else "—"}</td></tr>')

    @staticmethod
    def _trace_detail(item: dict[str, Any]) -> str:
        return " · ".join(f"{key}={value}" for key, value in item.items() if key != "event")[:240]

    @staticmethod
    def _score(value: Any) -> str:
        return f"{float(value):.3f}" if value is not None else "—"

    @staticmethod
    def _badge(status: str) -> str:
        color = STATUS_COLORS.get(status, "#8b9bb4")
        return f'<span class="badge" style="--badge:{color}">{escape(status)}</span>'

    @staticmethod
    def _page(title: str, content: str) -> str:
        return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>
<style>
:root{{--bg:#07111f;--panel:#0d1b2d;--line:#20334c;--text:#eff6ff;--muted:#8da2bd;--accent:#5b8cff}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 80% 0,#142b4b 0,transparent 35%),var(--bg);color:var(--text);font:14px Inter,system-ui,sans-serif}}
main{{width:min(1200px,calc(100% - 32px));margin:0 auto;padding:42px 0 70px}}header{{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;margin-bottom:30px}}h1{{font-size:clamp(26px,4vw,46px);margin:5px 0 8px;line-height:1.1}}h2{{margin:0;font-size:18px}}p{{color:var(--muted)}}.eyebrow{{color:#70a2ff;letter-spacing:.18em;font-size:11px;font-weight:700}}a{{color:#8fb3ff;text-decoration:none}}.api,nav a{{display:inline-block;border:1px solid var(--line);padding:9px 13px;border-radius:9px}}nav{{margin-bottom:24px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}}.metric,.panel{{background:linear-gradient(145deg,#102239,#0b1727);border:1px solid var(--line);border-radius:14px}}.metric{{padding:18px}}.metric span,.format-card span,small{{display:block;color:var(--muted);font-size:12px}}.metric strong{{font-size:28px;display:block;margin-top:8px}}.panel{{padding:20px;margin-top:16px}}.panel-title{{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;color:var(--muted)}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:13px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}}th{{color:var(--muted);font-size:11px;text-transform:uppercase}}td:first-child{{white-space:normal;min-width:220px}}small{{margin-top:4px}}.badge{{display:inline-block;border:1px solid var(--badge);color:var(--badge);background:color-mix(in srgb,var(--badge) 12%,transparent);padding:4px 8px;border-radius:999px;font-size:11px}}.formats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}}.format-card{{overflow:hidden;background:#081422;border:1px solid var(--line);border-radius:12px}}.format-card img{{width:100%;height:210px;object-fit:contain;background:#050b13}}.format-card div{{padding:12px;display:grid;gap:5px}}.trace{{list-style:none;padding:0;margin:0}}.trace li{{display:grid;grid-template-columns:220px 1fr;gap:12px;padding:10px 0;border-bottom:1px solid var(--line)}}code{{color:#83aaff}}.empty{{color:var(--muted);text-align:center;padding:28px}}
@media(max-width:720px){{.metrics{{grid-template-columns:repeat(2,1fr)}}header{{display:block}}.trace li{{grid-template-columns:1fr}}}}
</style></head><body><main>{content}</main></body></html>'''

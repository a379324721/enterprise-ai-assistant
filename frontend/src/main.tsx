import React, {FormEvent, useMemo, useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";

type Task = {id: string; title: string; operation: string; status: string; required_capabilities: string[]};
type Confirmation = {summary: string; action: string; payload: Record<string, unknown>};
type Result = {
  conversation_id: string; status: string; answer: string; user_goal: string;
  tasks: Task[]; slots: Record<string, unknown>; pending_confirmation?: Confirmation | null;
};
type Message = {role: "user" | "assistant"; text: string};

const examples = ["下周去上海出差，帮我申请，回来提醒报销", "我还有多少年假？下周五请一天年假", "查询差旅住宿标准"];

async function readResponse(response: Response): Promise<Result> {
  const body = await response.text();
  if (!body) {
    throw new Error(response.ok ? "服务返回了空响应" : `后端连接失败（HTTP ${response.status}）`);
  }
  let data: Result & {detail?: string};
  try {
    data = JSON.parse(body) as Result & {detail?: string};
  } catch {
    throw new Error(`服务返回了无法解析的响应（HTTP ${response.status}）`);
  }
  if (!response.ok) throw new Error(data.detail || `请求失败（HTTP ${response.status}）`);
  return data;
}

function App() {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const userId = useMemo(() => "demo-user", []);

  async function send(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || busy) return;
    const text = input.trim(); setInput(""); setBusy(true);
    setMessages((old) => [...old, {role: "user", text}]);
    try {
      const response = await fetch("/api/v1/chat", {method: "POST", headers: {"Content-Type": "application/json", "X-User-ID": userId}, body: JSON.stringify({message: text})});
      const data = await readResponse(response);
      setResult(data); setMessages((old) => [...old, {role: "assistant", text: data.answer || "任务已规划，请查看右侧执行状态。"}]);
    } catch (error) { setMessages((old) => [...old, {role: "assistant", text: error instanceof Error ? error.message : "系统异常"}]); }
    finally { setBusy(false); }
  }

  async function confirm(approved: boolean) {
    if (!result) return; setBusy(true);
    try {
      const response = await fetch(`/api/v1/conversations/${result.conversation_id}/confirm`, {method: "POST", headers: {"Content-Type": "application/json", "X-User-ID": userId}, body: JSON.stringify({approved})});
      const data = await readResponse(response);
      setResult(data); setMessages((old) => [...old, {role: "assistant", text: data.answer}]);
    } catch (error) { setMessages((old) => [...old, {role: "assistant", text: error instanceof Error ? error.message : "系统异常"}]); }
    finally { setBusy(false); }
  }

  return <main>
    <header><div className="brandMark">E</div><div><h1>Enterprise AI Assistant</h1><p>企业事务，一个对话完成</p></div><span className="online">● 系统在线</span></header>
    <section className="layout">
      <div className="chatPanel">
        <div className="intro"><span>AI</span><div><strong>你好，我是企业智能助手</strong><p>我可以协助差旅、报销、请假和制度查询。涉及提交的操作会先请你确认。</p></div></div>
        {messages.length === 0 && <div className="examples">{examples.map((item) => <button key={item} onClick={() => setInput(item)}>{item}<b>↗</b></button>)}</div>}
        <div className="messages">{messages.map((message, index) => <div key={index} className={`message ${message.role}`}>{message.text}</div>)}{busy && <div className="thinking">正在理解并规划任务…</div>}</div>
        {result?.pending_confirmation && <div className="confirmCard"><div className="risk">需要你的确认</div><strong>{result.pending_confirmation.summary}</strong><p>系统只会在你确认后执行该操作。</p><div><button className="cancel" onClick={() => confirm(false)}>取消</button><button className="approve" onClick={() => confirm(true)}>确认执行</button></div></div>}
        <form onSubmit={send}><textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="描述你想办理的事情…" rows={2}/><button disabled={busy}>发送</button></form>
      </div>
      <aside><div className="asideHead"><span>任务执行</span><small>{result ? `${result.tasks.filter(t => t.status === "completed").length}/${result.tasks.length}` : "0/0"}</small></div>
        {!result && <div className="empty"><i>⌁</i><p>发送请求后，这里会展示 AI 拆解出的任务及执行进度。</p></div>}
        {result && <><div className="goal"><small>理解到的目标</small><p>{result.user_goal}</p></div><div className="taskList">{result.tasks.map((task, index) => <div className="task" key={task.id}><span className={task.status}>{task.status === "completed" ? "✓" : index + 1}</span><div><strong>{task.title}</strong><small>{task.required_capabilities.join(" · ")}</small></div><em>{({completed:"已完成",running:"执行中",waiting_confirmation:"待确认",pending:"等待中",rejected:"已取消"} as Record<string,string>)[task.status] || task.status}</em></div>)}</div></>}
      </aside>
    </section>
  </main>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App/></React.StrictMode>);

import React, {FormEvent, useMemo, useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";
import "./streaming.css";

type Task = {id: string; title: string; domain: string; objective: string; status: string};
type Confirmation = {summary: string; action: string; payload: Record<string, unknown>};
type Result = {
  conversation_id: string; status: string; answer: string; user_goal: string;
  tasks: Task[]; artifacts: Record<string, unknown>; pending_confirmation?: Confirmation | null;
};
type Message = {role: "user" | "assistant"; text: string};
type SseMessage = {event: string; data: unknown};

const examples = ["下周去上海出差，帮我申请，回来提醒报销", "我还有多少年假？下周五请一天年假", "查询差旅住宿标准"];

async function consumeSse(
  response: Response,
  onEvent: (message: SseMessage) => void,
): Promise<void> {
  if (!response.ok) {
    const body = await response.text();
    let detail = "";
    try {
      detail = (JSON.parse(body) as {detail?: string}).detail ?? "";
    } catch { /* 代理可能返回纯文本或空响应体。 */ }
    throw new Error(detail || body || `流式请求失败（HTTP ${response.status}）`);
  }
  if (!response.body) throw new Error("浏览器没有收到可读取的响应流");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const {done, value} = await reader.read();
    buffer += decoder.decode(value, {stream: !done});
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      let event = "message";
      const dataLines: string[] = [];
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }
      if (!dataLines.length) continue;
      onEvent({event, data: JSON.parse(dataLines.join("\n")) as unknown});
    }
    if (done) break;
  }
}

function App() {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [progress, setProgress] = useState("");
  const userId = useMemo(() => "demo-user", []);

  function handleStreamEvent({event, data}: SseMessage) {
    if (event === "progress") {
      setProgress((data as {message: string}).message);
    } else if (event === "token") {
      const token = (data as {content: string}).content;
      setMessages((old) => old.map((message, index) => index === old.length - 1 ? {...message, text: message.text + token} : message));
    } else if (event === "done") {
      const completed = data as Result;
      setResult(completed);
      setMessages((old) => old.map((message, index) => index === old.length - 1 && !message.text ? {...message, text: completed.answer || "任务已完成。"} : message));
    } else if (event === "error") {
      throw new Error((data as {message: string}).message);
    }
  }

  async function send(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || busy) return;
    const text = input.trim(); setInput(""); setBusy(true); setProgress("正在连接智能助手");
    setMessages((old) => [...old, {role: "user", text}, {role: "assistant", text: ""}]);
    try {
      const response = await fetch("/api/v1/chat/stream", {method: "POST", headers: {"Content-Type": "application/json", "X-User-ID": userId}, body: JSON.stringify({message: text})});
      await consumeSse(response, handleStreamEvent);
    } catch (error) {
      const message = error instanceof Error ? error.message : "系统异常";
      setMessages((old) => old.map((item, index) => index === old.length - 1 ? {...item, text: item.text || message} : item));
    } finally { setBusy(false); setProgress(""); }
  }

  async function confirm(approved: boolean) {
    if (!result || busy) return;
    setBusy(true); setProgress(approved ? "正在确认并恢复任务" : "正在取消操作");
    setMessages((old) => [...old, {role: "assistant", text: ""}]);
    try {
      const response = await fetch(`/api/v1/conversations/${result.conversation_id}/confirm/stream`, {method: "POST", headers: {"Content-Type": "application/json", "X-User-ID": userId}, body: JSON.stringify({approved})});
      await consumeSse(response, handleStreamEvent);
    } catch (error) {
      const message = error instanceof Error ? error.message : "系统异常";
      setMessages((old) => old.map((item, index) => index === old.length - 1 ? {...item, text: item.text || message} : item));
    } finally { setBusy(false); setProgress(""); }
  }

  return <main>
    <header><div className="brandMark">E</div><div><h1>Enterprise AI Assistant</h1><p>企业事务，一个对话完成</p></div><span className="online">● 系统在线</span></header>
    <section className="layout">
      <div className="chatPanel">
        <div className="intro"><span>AI</span><div><strong>你好，我是企业智能助手</strong><p>我可以协助差旅、报销、请假和制度查询。涉及提交的操作会先请你确认。</p></div></div>
        {messages.length === 0 && <div className="examples">{examples.map((item) => <button key={item} onClick={() => setInput(item)}>{item}<b>↗</b></button>)}</div>}
        <div className="messages">{messages.map((message, index) => <div key={index} className={`message ${message.role}`}>{message.text}{busy && index === messages.length - 1 && message.role === "assistant" && <span className="cursor"/>}</div>)}{busy && <div className="thinking">{progress || "正在处理…"}</div>}</div>
        {result?.pending_confirmation && <div className="confirmCard"><div className="risk">需要你的确认</div><strong>{result.pending_confirmation.summary}</strong><p>系统只会在你确认后执行该操作。</p><div><button className="cancel" disabled={busy} onClick={() => confirm(false)}>取消</button><button className="approve" disabled={busy} onClick={() => confirm(true)}>确认执行</button></div></div>}
        <form onSubmit={send}><textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="描述你想办理的事情…" rows={2}/><button disabled={busy}>发送</button></form>
      </div>
      <aside><div className="asideHead"><span>任务执行</span><small>{result ? `${result.tasks.filter(t => t.status === "completed").length}/${result.tasks.length}` : "0/0"}</small></div>
        {!result && <div className="empty"><i>⌁</i><p>发送请求后，这里会展示 AI 拆解出的任务及执行进度。</p></div>}
        {result && <><div className="goal"><small>理解到的目标</small><p>{result.user_goal}</p></div><div className="taskList">{result.tasks.map((task, index) => <div className="task" key={task.id}><span className={task.status}>{task.status === "completed" ? "✓" : index + 1}</span><div><strong>{task.title}</strong><small>{task.domain}</small></div><em>{({completed:"已完成",running:"执行中",waiting_confirmation:"待确认",waiting_input:"待补充",pending:"等待中",rejected:"已取消"} as Record<string,string>)[task.status] || task.status}</em></div>)}</div></>}
      </aside>
    </section>
  </main>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App/></React.StrictMode>);

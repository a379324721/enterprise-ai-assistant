import {FormEvent, useEffect, useMemo, useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";
import "./streaming.css";

type Task = {id: string; title: string; operation: string; status: string; required_capabilities: string[]};
type Confirmation = {summary: string; action: string; payload: Record<string, unknown>};
type Result = {
  conversation_id: string; status: string; answer: string; user_goal: string;
  tasks: Task[]; task_history: TaskRun[]; messages: Message[];
  slots: Record<string, unknown>; pending_confirmation?: Confirmation | null;
};
type Message = {role: "user" | "assistant"; text: string};
type SseMessage = {event: string; data: unknown};
type TaskRun = {user_goal: string; tasks: Task[]};
type ConversationSummary = {conversation_id: string; title: string; created_at: string; updated_at: string};

const examples = ["下周去上海出差，帮我申请，回来提醒报销", "我还有多少年假？下周五请一天年假", "查询差旅住宿标准"];
const conversationStorageKey = "enterprise-assistant.current-conversation";
const capabilityLabels: Record<string, string> = {
  "travel.policy.read": "查询差旅制度",
  "travel.application.write": "提交差旅申请",
  "expense.policy.read": "查询报销制度",
  "expense.claim.write": "提交报销单",
  "expense.reminder.write": "设置报销提醒",
  "hr.leave.read": "查询休假信息",
  "hr.leave.write": "提交请假申请",
  "policy.search": "查询企业制度",
};

function formatConversationTime(value: string): string {
  return new Date(value).toLocaleString("zh-CN", {month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit"});
}

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
  const savedConversationId = useMemo(() => localStorage.getItem(conversationStorageKey), []);
  const [conversationId, setConversationId] = useState<string | null>(savedConversationId);
  const [taskHistory, setTaskHistory] = useState<TaskRun[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const userId = useMemo(() => "demo-user", []);

  function rememberConversation(id: string) {
    localStorage.setItem(conversationStorageKey, id);
    setConversationId(id);
  }

  function applyConversation(restored: Result) {
    rememberConversation(restored.conversation_id);
    setResult(restored);
    setTaskHistory(restored.task_history);
    setMessages(restored.messages);
  }

  async function refreshConversationList(signal?: AbortSignal) {
    const response = await fetch("/api/v1/conversations", {
      headers: {"X-User-ID": userId},
      signal,
    });
    if (!response.ok) throw new Error(`加载历史会话失败（HTTP ${response.status}）`);
    const items = await response.json() as ConversationSummary[];
    setConversations(items);
    return items;
  }

  async function fetchConversation(id: string, signal?: AbortSignal) {
    const response = await fetch(`/api/v1/conversations/${id}`, {
      headers: {"X-User-ID": userId},
      signal,
    });
    if (!response.ok) throw new Error(`加载会话失败（HTTP ${response.status}）`);
    return response.json() as Promise<Result>;
  }

  useEffect(() => {
    const controller = new AbortController();
    setBusy(true);
    setProgress("正在恢复历史会话");
    void (async () => {
      let restoredSavedConversation = false;
      if (savedConversationId) {
        try {
          applyConversation(await fetchConversation(savedConversationId, controller.signal));
          restoredSavedConversation = true;
        } catch (error) {
          if (error instanceof DOMException && error.name === "AbortError") throw error;
          localStorage.removeItem(conversationStorageKey);
          setConversationId(null);
        }
      }
      const items = await refreshConversationList(controller.signal);
      if (!restoredSavedConversation && items[0]) {
        applyConversation(await fetchConversation(items[0].conversation_id, controller.signal));
      }
    })().catch((error: unknown) => {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setMessages([{role: "assistant", text: error instanceof Error ? error.message : "恢复历史会话失败"}]);
    }).finally(() => {
      if (!controller.signal.aborted) {
        setBusy(false);
        setProgress("");
      }
    });
    return () => controller.abort();
  }, [savedConversationId, userId]);

  async function selectConversation(id: string) {
    if (busy || id === conversationId) return;
    setBusy(true); setProgress("正在加载历史会话");
    try {
      applyConversation(await fetchConversation(id));
    } catch (error) {
      setMessages([{role: "assistant", text: error instanceof Error ? error.message : "加载会话失败"}]);
    } finally { setBusy(false); setProgress(""); }
  }

  function startNewConversation() {
    if (busy) return;
    localStorage.removeItem(conversationStorageKey);
    setConversationId(null);
    setResult(null);
    setTaskHistory([]);
    setMessages([]);
    setInput("");
  }

  function handleStreamEvent({event, data}: SseMessage) {
    if (event === "metadata") {
      rememberConversation((data as {conversation_id: string}).conversation_id);
    } else if (event === "progress") {
      setProgress((data as {message: string}).message);
    } else if (event === "token") {
      const token = (data as {content: string}).content;
      setMessages((old) => old.map((message, index) => index === old.length - 1 ? {...message, text: message.text + token} : message));
    } else if (event === "done") {
      const completed = data as Result;
      rememberConversation(completed.conversation_id);
      setResult(completed);
      setTaskHistory(completed.task_history);
      void refreshConversationList();
      setMessages((old) => old.map((message, index) => index === old.length - 1 && !message.text ? {...message, text: completed.answer || "暂时无法生成回答，请换一种方式描述。"} : message));
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
      const response = await fetch("/api/v1/chat/stream", {method: "POST", headers: {"Content-Type": "application/json", "X-User-ID": userId}, body: JSON.stringify({message: text, ...(conversationId ? {conversation_id: conversationId} : {})})});
      await consumeSse(response, handleStreamEvent);
    } catch (error) {
      const message = error instanceof Error ? error.message : "系统异常";
      if (conversationId) {
        try {
          applyConversation(await fetchConversation(conversationId));
          setMessages((old) => [...old, {role: "assistant", text: message}]);
        } catch {
          setMessages((old) => old.map((item, index) => index === old.length - 1 ? {...item, text: item.text || message} : item));
        }
      } else {
        setMessages((old) => old.map((item, index) => index === old.length - 1 ? {...item, text: item.text || message} : item));
      }
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
      try {
        applyConversation(await fetchConversation(result.conversation_id));
        setMessages((old) => [...old, {role: "assistant", text: message}]);
      } catch {
        setMessages((old) => old.map((item, index) => index === old.length - 1 ? {...item, text: item.text || message} : item));
      }
    } finally { setBusy(false); setProgress(""); }
  }

  const visibleTaskRuns = [
    ...taskHistory,
    ...(result?.tasks.length ? [{user_goal: result.user_goal, tasks: result.tasks}] : []),
  ];

  return <main>
    <header><div className="brandMark">E</div><div><h1>Enterprise AI Assistant</h1><p>企业事务，一个对话完成</p></div><span className="online">● 系统在线</span></header>
    <section className="layout">
      <nav className="historyPanel"><div className="historyHead"><strong>历史会话</strong><button disabled={busy} onClick={startNewConversation}>＋ 新建</button></div><div className="conversationList">{conversations.length === 0 && <p className="noConversations">暂无历史会话</p>}{conversations.map((conversation) => <button className={conversation.conversation_id === conversationId ? "active" : ""} disabled={busy} key={conversation.conversation_id} onClick={() => selectConversation(conversation.conversation_id)}><strong>{conversation.title}</strong><small>{formatConversationTime(conversation.updated_at)}</small></button>)}</div></nav>
      <div className="chatPanel">
        <div className="intro"><span>AI</span><div><strong>你好，我是企业智能助手</strong><p>我可以协助差旅、报销、请假和制度查询。涉及提交的操作会先请你确认。</p></div></div>
        {messages.length === 0 && <div className="examples">{examples.map((item) => <button key={item} onClick={() => setInput(item)}>{item}<b>↗</b></button>)}</div>}
        <div className="messages">{messages.map((message, index) => <div key={index} className={`message ${message.role}`}>{message.text}{busy && index === messages.length - 1 && message.role === "assistant" && <span className="cursor"/>}</div>)}{busy && <div className="thinking">{progress || "正在处理…"}</div>}</div>
        {result?.pending_confirmation && <div className="confirmCard"><div className="risk">需要你的确认</div><strong>{result.pending_confirmation.summary}</strong><p>系统只会在你确认后执行该操作。</p><div><button className="cancel" disabled={busy} onClick={() => confirm(false)}>取消</button><button className="approve" disabled={busy} onClick={() => confirm(true)}>确认执行</button></div></div>}
        <form onSubmit={send}><textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="描述你想办理的事情…" rows={2}/><button disabled={busy}>发送</button></form>
      </div>
      <aside><div className="asideHead"><span>任务执行</span>{result && result.tasks.length > 0 && <small>{`当前 ${result.tasks.filter(task => task.status === "completed").length}/${result.tasks.length}`}</small>}</div>
        {visibleTaskRuns.length === 0 && <div className="empty"><i>⌁</i><p>发送业务请求后，这里会展示 AI 拆解出的任务及执行进度。</p></div>}
        {visibleTaskRuns.map((run, runIndex) => <div className="taskRun" key={`${runIndex}-${run.user_goal}`}><div className="goal"><small>{runIndex < taskHistory.length ? "历史目标" : "当前目标"}</small><p>{run.user_goal}</p></div><div className="taskList">{run.tasks.map((task, taskIndex) => <div className="task" key={`${runIndex}-${task.id}`}><span className={task.status}>{task.status === "completed" ? "✓" : taskIndex + 1}</span><div><strong>{task.title}</strong><small>{task.required_capabilities.map(capability => capabilityLabels[capability] || capability).join(" · ")}</small></div><em>{({completed:"已完成",running:"执行中",waiting_input:"待补充",waiting_confirmation:"待确认",pending:"等待中",rejected:"已取消",failed:"执行失败"} as Record<string,string>)[task.status] || task.status}</em></div>)}</div></div>)}
      </aside>
    </section>
  </main>;
}

createRoot(document.getElementById("root")!).render(<App/>);

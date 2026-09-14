import React, {useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState} from "react";
import {createRoot} from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./styles.css";
import "./streaming.css";

type Task = {id: string; title: string; domain: string; objective: string; status: string};
type Confirmation = {confirmation_id: string; title: string; action: string; fields: {name: string; label: string; value: string}[]; payload: Record<string, unknown>};
type DraftField = {name: string; label: string; value: string; source: "user" | "memory" | "dependency"};
//: 一件还没办完的事。只有卡在待补充、待确认上的计划才会成为事项，由后端投影。
type Matter = {
  plan_id: string; status: "waiting_input" | "waiting_confirmation" | "shelved";
  task_id: string; title: string; known_fields: DraftField[]; missing_fields: string[];
  tasks: {id: string; title: string; domain: string; status: string}[];
};
type TurnStep = {id: string; task_id: string; label: string; success: boolean};
type Result = {
  conversation_id: string; status: string; answer: string; user_goal: string;
  tasks: Task[]; artifacts: Record<string, unknown>; pending_confirmation?: Confirmation | null;
  matters: Matter[]; steps: TurnStep[];
};
type ActionItem = {reference_id: string; action_type: string; summary: string; created_at: string; fields: Record<string, string>; revoked_at: string | null};
//: decision 不是对话双方说的话，而是用户在确认卡片上做的选择。后端把它作为一条
//: SystemMessage 追加到会话历史，所以刷新后仍在；本地这条只是为了立刻有反馈。
//: steps 是这一轮执行过的工具调用，只在本地插入，不进会话历史——刷新后不再显示。
type Message = {
  role: "user" | "assistant" | "decision" | "steps"; text: string; index?: number;
  steps?: TurnStep[];
  // 这段回答属于哪个任务。流式期间只有 answer_start 带来的领域名可用，done 之后
  // 能在 result.tasks 里换到真正的任务标题。历史消息没有这两个字段。
  taskId?: string; agent?: string;
};
//: 延迟渲染那一轮攒下的回答段落，一段对应一个任务。
type AnswerSegment = {taskId?: string; agent?: string; text: string};
type SseMessage = {event: string; data: unknown};
type Session = {token: string; userId: string; displayName: string; conversationId: string};

const examples = ["申请后天去上海出差，顺便订个当天上午的会议室", "我还有多少年假？下周五请一天年假", "查询差旅住宿标准"];
//: 领域名的中文标签，用于多任务回答的分节标题。
const DOMAIN_LABELS: Record<string, string> = {
  travel: "差旅",
  expense: "报销",
  hr: "人事",
  meeting: "会议室",
  policy: "制度",
};

//: 单据类型的中文标签。字段本身不翻译——白名单在后端
//: （_ACTION_SUMMARY_FIELDS），在前端再抄一份字段名迟早会漂移。
const ACTION_LABELS: Record<string, string> = {
  travel_application: "差旅申请",
  expense_claim: "费用报销",
  leave_request: "请假申请",
  meeting_booking: "会议室",
};

//: 单据字段里的枚举值。字段名和顺序仍由后端白名单决定，这里只翻译取值。
const VALUE_LABELS: Record<string, string> = {
  one_way: "单程",
  round_trip: "往返",
};

const TASK_STATUS_LABELS: Record<string, string> = {
  completed: "已完成", running: "执行中", waiting_confirmation: "待确认",
  waiting_input: "待补充", pending: "等待中", rejected: "已取消", failed: "失败",
};

const MATTER_STATUS_LABELS: Record<Matter["status"], string> = {
  waiting_input: "待补充",
  waiting_confirmation: "待确认",
  shelved: "已搁置",
};

const SESSION_KEY = "eaa.session";
const PAGE_SIZE = 20;

function readSession(): Session | null {
  try {
    const raw = window.localStorage.getItem(SESSION_KEY);
    return raw ? JSON.parse(raw) as Session : null;
  } catch { return null; }
}

function MarkdownMessage({text}: {text: string}) {
  return <ReactMarkdown
    remarkPlugins={[remarkGfm]}
    components={{
      a: ({href, children}) => <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>,
    }}
  >{text}</ReactMarkdown>;
}

/**
 * 取服务端的 detail 作为提示。
 *
 * FastAPI 的校验失败（422）返回的 detail 是一个数组而不是字符串，直接当字符串用
 * 会在界面上渲染成 [object Object]，所以这里按两种形状分别取。
 */
async function readError(response: Response, fallback: string): Promise<string> {
  const body = await response.text();
  let detail: unknown;
  try {
    detail = (JSON.parse(body) as {detail?: unknown}).detail;
  } catch { return body.trim() || fallback; }
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => (item as {msg?: string}).msg)
      .filter((item): item is string => Boolean(item));
    if (messages.length) return messages.join("；");
  }
  return fallback;
}

/** 把异常翻成用户看得懂的一句话。fetch 连不上服务端时抛 TypeError。 */
function describeFailure(issue: unknown, fallback = "加载失败"): string {
  if (issue instanceof TypeError) return "无法连接服务器，请确认后端已启动";
  return issue instanceof Error && issue.message ? issue.message : fallback;
}

async function consumeSse(
  response: Response,
  onEvent: (message: SseMessage) => void,
): Promise<void> {
  if (!response.ok) throw new Error(await readError(response, `流式请求失败（HTTP ${response.status}）`));
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

/** 演示登录：只有名字，没有凭据。正式部署由企业 SSO 取代这一屏。 */
function LoginView({onSignedIn}: {onSignedIn: (session: Session) => void}) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit() {
    const trimmed = name.trim();
    if (!trimmed || busy) return;
    setBusy(true); setError("");
    try {
      const response = await fetch("/api/v1/auth/login", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({name: trimmed}),
      });
      if (!response.ok) {
        // 名字不存在会被直接创建，所以 404 只剩一种含义：接口没开。
        throw new Error(await readError(
          response,
          response.status === 404
            ? "演示登录未开启，请确认服务端的 DEMO_LOGIN_ENABLED"
            : `进入失败（HTTP ${response.status}）`,
        ));
      }
      const body = await response.json() as {
        access_token: string; user_id: string; display_name: string; conversation_id: string;
      };
      const session: Session = {
        token: body.access_token,
        userId: body.user_id,
        displayName: body.display_name,
        conversationId: body.conversation_id,
      };
      window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
      onSignedIn(session);
    } catch (issue) {
      setError(describeFailure(issue, "进入失败"));
    } finally { setBusy(false); }
  }

  return <main className="loginPage">
    <div className="loginCard">
      <div className="brandMark">E</div>
      <h1>Enterprise AI Assistant</h1>
      <p>输入你的名字即可开始。演示环境不设密码，一个名字对应一个用户，下次用同一个名字会回到同一个会话。</p>
      <form onSubmit={(event) => { event.preventDefault(); void submit(); }}>
        <input value={name} onChange={(event) => setName(event.target.value)} placeholder="你的名字" autoFocus maxLength={64}/>
        <button className="approve" disabled={busy || !name.trim()}>{busy ? "正在进入…" : "进入"}</button>
      </form>
      {error && <p className="loginError">{error}</p>}
    </div>
  </main>;
}

function ChatView({session, onSignOut}: {session: Session; onSignOut: () => void}) {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [progress, setProgress] = useState("");
  const [hasMore, setHasMore] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [actions, setActions] = useState<ActionItem[]>([]);
  // 回答正在逐字出现时不必再挂一行"正在生成回答"——气泡末尾的光标已经说明了。
  const [streamingAnswer, setStreamingAnswer] = useState(false);
  const activeAnswerId = useRef<string | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  // 是否跟随底部。用 ref 而不是 state：它每次滚动都会变，进 state 会白白多渲染一轮。
  const stickToBottom = useRef(true);
  // 向上翻页前记下的"距底部距离"，用来在新内容插到上方后把视口钉回原处。
  const scrollAnchor = useRef<number | null>(null);
  // 确认那条路径不逐字渲染：增量丢掉，最终结果攒在 deferredResult 里等一起提交。
  const deferStream = useRef(false);
  const deferredResult = useRef<Result | null>(null);
  const deferredSegments = useRef<AnswerSegment[]>([]);
  // 确认那一轮里已经随 task_done 提前画出来的任务数。done 时没有剩下的段落，不代表
  // 回答没来过，不能再把 completed.answer 整段补一遍。
  const deferredCommitted = useRef(0);
  // 已经画进对话流的步骤。确认前后的两次 done 都带着同一轮更早的步骤，不去重会画两遍。
  const shownSteps = useRef<Set<string>>(new Set());
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  // 向上翻页会把更早的消息插到当前内容上方，浏览器只保留 scrollTop 的数值，于是
  // 视口相对内容整体上移——用户刚才在看的那条被推到屏幕外，看着就像"跳到最上面"。
  // 按距底部的距离还原，原来那条消息就还停在同一个位置。必须用 layout effect：
  // 放在 passive effect 里会先画出跳掉的一帧。
  //
  // 只能挂在 messages 上：请求发出时 setLoadingHistory 会先渲染一次（按钮变成
  // "加载中…"），那一帧内容高度还没变，无依赖的 effect 会在这里就把锚点消耗掉，
  // 等更早的消息真正插进来时已经没有锚点可用了。
  useLayoutEffect(() => {
    const box = messagesRef.current;
    const anchor = scrollAnchor.current;
    if (!box || anchor === null) return;
    scrollAnchor.current = null;
    box.scrollTop = box.scrollHeight - anchor;
    // 用户已经明确要往上看了。不关掉跟随，下面那个 effect 会紧接着把视口拽回底部
    // （消息本来不足一屏时 stickToBottom 仍是 true），补偿等于白做。
    stickToBottom.current = false;
  }, [messages]);

  // 没有依赖数组，每次渲染后都对齐一次底部：首屏历史、发送、流式增量、确认卡片
  // 出现都会改变内容高度，逐个列依赖容易漏掉一种，而这里的代价只是一次赋值。
  // 只在用户本来就贴着底部时才跟随，否则往上翻历史会被新来的增量硬拽回去。
  useEffect(() => {
    const box = messagesRef.current;
    if (!box || !stickToBottom.current) return;
    box.scrollTop = box.scrollHeight;
  });

  const authHeaders = useMemo(
    () => ({"Content-Type": "application/json", Authorization: `Bearer ${session.token}`}),
    [session.token],
  );

  // 令牌过期后留在界面上只会不断报错，直接退回登录页重新取。
  const guard = useCallback((response: Response) => {
    if (response.status === 401) { onSignOut(); throw new Error("登录已过期，请重新登录"); }
    return response;
  }, [onSignOut]);

  /** 取最近提交过的单据，由调用方决定什么时候落到界面上——确认那条路径要等它和
   *  回答、任务面板一起提交。失败返回 null：单据面板是旁路信息，不该把对话拖下水。 */
  const fetchActions = useCallback(async (signal?: AbortSignal): Promise<ActionItem[] | null> => {
    try {
      const response = await fetch("/api/v1/actions?limit=10", {headers: authHeaders, signal});
      if (!response.ok) return null;
      return (await response.json() as {actions: ActionItem[]}).actions;
    } catch { return null; }
  }, [authHeaders]);

  /** 取一页历史，返回本页条数，供调用方判断会话是否已经有内容。 */
  const loadHistory = useCallback(async (before?: number, signal?: AbortSignal) => {
    const query = new URLSearchParams({limit: String(PAGE_SIZE)});
    if (before !== undefined) query.set("before", String(before));
    const response = guard(await fetch(
      `/api/v1/conversations/${session.conversationId}/messages?${query}`,
      {headers: authHeaders, signal},
    ));
    if (!response.ok) return 0;
    const body = await response.json() as {messages: Required<Message>[]; has_more: boolean};
    setHasMore(body.has_more);
    setMessages((old) => before === undefined ? body.messages : [...body.messages, ...old]);
    return body.messages.length;
  }, [authHeaders, guard, session.conversationId]);

  // 进入会话时补齐最近一页历史，并把上一轮未完成的确认重新摆出来——
  // 演示时刷新页面不该让一个待确认的写操作凭空消失。
  const enterConversation = useCallback(async (signal?: AbortSignal) => {
    setLoadingHistory(true); setLoadError("");
    void fetchActions(signal).then((list) => list && setActions(list));
    try {
      const loaded = await loadHistory(undefined, signal);
      // 会话快照只用来恢复待确认操作和任务面板，历史为空时二者必然都不存在。
      // 新用户的会话尚未落检查点，这一请求只会换回一个 404。
      if (loaded > 0) {
        const snapshot = await fetch(
          `/api/v1/conversations/${session.conversationId}`,
          {headers: authHeaders, signal},
        );
        if (snapshot.ok) {
          const restored = await snapshot.json() as Result;
          // 快照里的步骤属于上一轮，而那一轮的消息来自历史，没有位置可以插回去。
          restored.steps.forEach((step) => shownSteps.current.add(step.id));
          setResult(restored);
        }
      }
    } catch (issue) {
      // 中止来自 effect 清理，不是故障；其余情况必须说出来，否则后端没起时
      // 界面只是一片空白，看起来像历史被清掉了。
      if (signal?.aborted) return;
      setLoadError(describeFailure(issue, "历史消息加载失败"));
    } finally {
      if (!signal?.aborted) setLoadingHistory(false);
    }
  }, [authHeaders, fetchActions, loadHistory, session.conversationId]);

  useEffect(() => {
    // StrictMode 在开发模式下会把 effect 跑两遍，切换用户也会重跑；没有中止信号的话，
    // 先发出的那次响应可能后到并覆盖新一次的结果。
    const controller = new AbortController();
    void enterConversation(controller.signal);
    return () => controller.abort();
  }, [enterConversation]);

  async function clearConversation() {
    if (busy || !window.confirm("清空当前会话的全部消息？该操作不可撤销。")) return;
    setBusy(true);
    try {
      const response = guard(await fetch(`/api/v1/conversations/${session.conversationId}`, {
        method: "DELETE", headers: authHeaders,
      }));
      if (!response.ok) throw new Error(await readError(response, "清空失败"));
      setMessages([]); setResult(null); setHasMore(false); setLoadError("");
      shownSteps.current.clear();
    } catch (issue) {
      setLoadError(describeFailure(issue, "清空失败"));
    } finally { setBusy(false); }
  }

  async function loadEarlier() {
    const earliest = messages.find((message) => message.index !== undefined)?.index;
    if (earliest === undefined || loadingHistory) return;
    setLoadingHistory(true);
    const box = messagesRef.current;
    scrollAnchor.current = box ? box.scrollHeight - box.scrollTop : null;
    try { await loadHistory(earliest); } finally { setLoadingHistory(false); }
  }

  /** 这一段回答属于哪个任务。
   *
   *  流式期间只有领域名（answer_start 带的 agent），done 之后 result.tasks 里
   *  能换到真正的任务标题。历史消息没有 task_id，返回空串即不显示标题。
   */
  function sectionLabel(message: Message): string {
    if (!message.taskId && !message.agent) return "";
    const task = result?.tasks.find((item) => item.id === message.taskId);
    return task?.title || DOMAIN_LABELS[message.agent ?? ""] || "";
  }

  /** 把一个任务的执行步骤插到它的回答前面：先做了什么，再说结果。
   *
   *  找不到这个任务的回答时（比如回答还没画出来），放在本轮已有内容的末尾、
   *  send() 预插的空占位之前，不能插到本轮开头去——那会排到前面任务的回答之前。
   */
  function insertSteps(current: Message[], steps: TurnStep[], taskId: string): Message[] {
    if (steps.length === 0) return current;
    let boundary = current.length - 1;
    while (boundary >= 0 && current[boundary].role !== "user" && current[boundary].role !== "decision") boundary -= 1;
    let at = current.findIndex((message, index) => index > boundary && message.role === "assistant" && message.taskId === taskId);
    if (at < 0) {
      const last = current.at(-1);
      at = last && last.role === "assistant" && !last.text ? current.length - 1 : current.length;
    }
    return [...current.slice(0, at), {role: "steps", text: "", steps}, ...current.slice(at)];
  }

  /** 把一轮的最终结果落到界面上。
   *
   *  actionList 传入时，回答、任务面板、单据在同一次渲染里出现；传 null 表示
   *  这一轮是流式的，单据补一次异步刷新就行——回答早就在屏幕上了。
   */
  function applyCompletion(
    completed: Result, actionList: ActionItem[] | null, segments?: AnswerSegment[],
  ) {
    const fresh = completed.steps.filter((step) => !shownSteps.current.has(step.id));
    fresh.forEach((step) => shownSteps.current.add(step.id));
    setMessages((current) => {
      // 步骤插在本轮用户那句话（或确认决定）之后、回答之前：先做了什么，再说结果。
      let old = current;
      if (fresh.length > 0) {
        let at = current.length - 1;
        while (at >= 0 && current[at].role !== "user" && current[at].role !== "decision") at -= 1;
        old = [...current.slice(0, at + 1), {role: "steps", text: "", steps: fresh}, ...current.slice(at + 1)];
      }
      if (segments) {
        // 延迟渲染的那一轮没有占位气泡（不然会是个挂着光标的空泡），攒下的段落
        // 直接追加，每段仍然带着自己的任务。
        const written = segments.filter((segment) => segment.text.trim());
        if (written.length > 0) {
          return [...old, ...written.map((segment) => ({
            role: "assistant" as const, text: segment.text,
            taskId: segment.taskId, agent: segment.agent,
          }))];
        }
        // 回答都已经随各自的 task_done 画出来了。
        if (deferredCommitted.current > 0) return old;
        // 一个 token 都没来过：可能是回答走了非流式路径，也可能又冒出一个待确认
        // （那时没有回答，卡片自己会出来）。
        if (completed.answer) return [...old, {role: "assistant", text: completed.answer}];
        if (completed.status === "waiting_confirmation") return old;
        return [...old, {role: "assistant", text: "未生成有效回复，请重试。"}];
      }
      const last = old.at(-1);
      if (!last || last.role !== "assistant" || last.text) return old;
      if (completed.answer) {
        return old.map((message, index) => index === old.length - 1 ? {...message, text: completed.answer} : message);
      }
      if (completed.status === "waiting_confirmation") return old.slice(0, -1);
      return old.map((message, index) => index === old.length - 1 ? {...message, text: "未生成有效回复，请重试。"} : message);
    });
    // 事项由后端从检查点投影，每一轮都是权威的全量，闲聊轮也照样覆盖。
    setResult(completed);
    // 没有调用过工具的轮次不可能新增单据，不必再拉一次。
    if (actionList) setActions(actionList);
    else if (completed.steps.length > 0) void fetchActions().then((list) => list && setActions(list));
  }

  function handleStreamEvent({event, data}: SseMessage) {
    if (event === "progress") {
      // 进度事件意味着换了节点，当前这段回答不再有新增量。
      setProgress((data as {message: string}).message);
      setStreamingAnswer(false);
    } else if (event === "answer_start") {
      const start = data as {message_id: string; agent?: string; task_id?: string};
      if (activeAnswerId.current === start.message_id) return;
      activeAnswerId.current = start.message_id;
      if (deferStream.current) {
        // 不渲染，但仍然按任务分段攒着，最后一次性提交时照样能分节。
        deferredSegments.current.push({taskId: start.task_id, agent: start.agent, text: ""});
        return;
      }
      // 每个任务的回答自成一条。挤进同一个气泡时，两段讲不同事情的话读起来像
      // 两个人在插话；分开之后它们和右侧任务面板一一对应。
      setMessages((old) => {
        const last = old.at(-1);
        // send() 预插的空占位留给第一段，后续任务各开一条。
        if (last && last.role === "assistant" && !last.text) {
          return old.map((message, index) => index === old.length - 1
            ? {...message, agent: start.agent, taskId: start.task_id} : message);
        }
        return [...old, {role: "assistant", text: "", agent: start.agent, taskId: start.task_id}];
      });
    } else if (event === "token") {
      const chunk = (data as {content: string}).content;
      if (deferStream.current) {
        const segment = deferredSegments.current.at(-1);
        if (segment) segment.text += chunk;
        else deferredSegments.current.push({text: chunk});
        return;
      }
      // 放在延迟分支之后：确认那一轮屏幕上没有逐字出现的回答，转圈得一直转着。
      setStreamingAnswer(true);
      setMessages((old) => old.map((message, index) => index === old.length - 1 ? {...message, text: message.text + chunk} : message));
    } else if (event === "task_done") {
      // 一个任务办完了。确认之后常常还有依赖它的任务要跑十几秒，不能让这个任务的步骤和
      // 回答陪着等到整轮结束：步骤连同回答按任务依次呈现。
      const finished = data as {task_id: string; steps: TurnStep[]};
      const fresh = finished.steps.filter((step) => !shownSteps.current.has(step.id));
      fresh.forEach((step) => shownSteps.current.add(step.id));
      if (deferStream.current) {
        const own = deferredSegments.current.filter((segment) => segment.taskId === finished.task_id && segment.text.trim());
        deferredSegments.current = deferredSegments.current.filter((segment) => segment.taskId !== finished.task_id);
        deferredCommitted.current += 1;
        setMessages((old) => [
          ...old,
          ...(fresh.length > 0 ? [{role: "steps" as const, text: "", steps: fresh}] : []),
          ...own.map((segment) => ({role: "assistant" as const, text: segment.text, taskId: segment.taskId, agent: segment.agent})),
        ]);
        // 单据和回答一起刷新，保留延迟渲染要的"三者同时出现"，只是粒度从整轮变成任务。
        if (fresh.length > 0) void fetchActions().then((list) => list && setActions(list));
        return;
      }
      if (fresh.length > 0) setMessages((old) => insertSteps(old, fresh, finished.task_id));
    } else if (event === "done") {
      const completed = data as Result;
      // 延迟模式下这一轮什么都还没画出来，交给 confirm 连同单据一次性提交。
      if (deferStream.current) { deferredResult.current = completed; return; }
      applyCompletion(completed, null);
    } else if (event === "error") {
      throw new Error((data as {message: string}).message);
    }
  }

  async function send(preset?: string) {
    const text = (preset ?? input).trim();
    if (!text || busy) return;
    // 点示例时输入框里可能有用户打了一半的草稿，不替他清掉。
    if (preset === undefined) setInput("");
    setBusy(true); setProgress("正在连接智能助手");
    // 自己发出的消息一定要看见，哪怕此刻正停在历史里翻看。
    stickToBottom.current = true;
    setStreamingAnswer(false);
    setMessages((old) => [...old, {role: "user", text}, {role: "assistant", text: ""}]);
    try {
      const response = guard(await fetch("/api/v1/chat/stream", {
        method: "POST", headers: authHeaders,
        body: JSON.stringify({message: text, request_id: crypto.randomUUID(), conversation_id: session.conversationId}),
      }));
      await consumeSse(response, handleStreamEvent);
    } catch (issue) {
      const message = describeFailure(issue, "系统异常");
      setMessages((old) => old.map((item, index) => index === old.length - 1 ? {...item, text: item.text || message} : item));
    } finally {
      setBusy(false); setProgress(""); setStreamingAnswer(false);
      activeAnswerId.current = null;
    }
  }

  async function confirm(approved: boolean) {
    const pending = result?.pending_confirmation;
    if (!pending || busy) return;
    setBusy(true); setProgress(approved ? "正在确认并恢复任务" : "正在取消操作");
    // 先本地插一条，不等后端往返。文案与 routes.py 的 _decision_message 一致，
    // 刷新后从历史里取回的是同一句，看不出差别。
    setMessages((old) => [
      ...old,
      {role: "decision", text: `${approved ? "你确认了" : "你取消了"}：${pending.title}`},
    ]);
    // 决定已经做出，卡片不必再挂着等图跑完。等 done 事件才收的话，取消之后那几秒
    // 里卡片还在原地，看着像根本没点上。权威状态随后由 done 事件覆盖。
    setResult((old) => old ? {...old, pending_confirmation: null} : old);
    // 这一轮不逐字渲染：确认之后真正要看的是任务状态和单据，回答只有一两句。
    // 逐字出字会让三者错开好几秒——回答先到，任务面板后到，单据再后到。
    deferStream.current = true; deferredResult.current = null; deferredSegments.current = [];
    deferredCommitted.current = 0;
    try {
      const response = guard(await fetch(`/api/v1/conversations/${session.conversationId}/confirm/stream`, {
        method: "POST", headers: authHeaders,
        body: JSON.stringify({confirmation_id: pending.confirmation_id, approved}),
      }));
      await consumeSse(response, handleStreamEvent);
      const completed = deferredResult.current;
      // 单据先取回来，再和回答、任务面板一次性提交：React 会把这几个 setState
      // 合成一次渲染，三者同一帧出现，转圈在此之前一直转着。
      if (completed) applyCompletion(completed, await fetchActions(), deferredSegments.current);
    } catch (issue) {
      setMessages((old) => [...old, {role: "assistant", text: describeFailure(issue, "系统异常")}]);
      // 请求没走通，服务端那边仍然停在等确认上；卡片必须放回去，否则用户再没有入口。
      setResult((old) => old && !old.pending_confirmation ? {...old, pending_confirmation: pending} : old);
    } finally {
      setBusy(false); setProgress(""); setStreamingAnswer(false);
      activeAnswerId.current = null;
      deferStream.current = false; deferredResult.current = null; deferredSegments.current = [];
    }
  }

  const matters = result?.matters ?? [];

  return <main>
    <header><div className="brandMark">E</div><div><h1>Enterprise AI Assistant</h1><p>企业事务，一个对话完成</p></div>
      <span className="online">● {session.displayName}</span>
      <button className="signOut" disabled={busy} onClick={() => void clearConversation()}>清空会话</button>
      <button className="signOut" onClick={onSignOut}>退出</button>
    </header>
    <section className="layout">
      <div className="chatPanel">
        <div className="intro"><span>AI</span><div><strong>我是企业智能助手</strong><p>我可以协助差旅、报销、请假、会议室预订和制度查询，也能帮你查已提交单据的状态。涉及提交的操作会先请你确认。</p></div></div>
        {messages.length === 0 && !loadingHistory && <div className="examples">{examples.map((item) => <button key={item} disabled={busy} onClick={() => void send(item)}>{item}<b>↗</b></button>)}</div>}
        {loadError && <div className="loadBanner"><span>{loadError}</span><button disabled={loadingHistory} onClick={() => void enterConversation()}>重试</button></div>}
        <div
          className="messages"
          ref={messagesRef}
          onScroll={(event) => {
            const box = event.currentTarget;
            // 留 40px 容差：流式输出时行高不断变化，要求严格贴底会被误判成用户已经离开底部。
            stickToBottom.current = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
          }}
        >
          {hasMore && <button className="loadEarlier" disabled={loadingHistory} onClick={() => void loadEarlier()}>{loadingHistory ? "加载中…" : "加载更早的消息"}</button>}
          {messages.map((message, index) => <div key={message.index ?? `live-${index}`} className={`message ${message.role}`}>
            {message.role === "steps" && message.steps?.map((step) => <span key={step.id} className={step.success ? "step ok" : "step fail"}>{step.success ? "✓" : "✕"} {step.label}</span>)}
            {sectionLabel(message) && <span className="section">{sectionLabel(message)}</span>}
            {message.role === "assistant" ? <MarkdownMessage text={message.text}/> : message.role === "steps" ? null : message.text}
            {busy && index === messages.length - 1 && message.role === "assistant" && <span className="cursor"/>}
          </div>)}
          {busy && !streamingAnswer && <div className="thinking"><span className="spinner"/>{progress || "正在处理…"}</div>}
        </div>
        {result?.pending_confirmation && <div className="confirmCard"><div className="risk">需要你的确认</div><strong>{result.pending_confirmation.title}</strong>
          {/* 字段名和取值标签由后端按工具入参契约给出，这里只负责排版。 */}
          <dl className="confirmFields">{result.pending_confirmation.fields.map((field) => <div key={field.name}><dt>{field.label}</dt><dd>{field.value}</dd></div>)}</dl>
          <p>系统只会在你确认后执行该操作。</p><div><button className="cancel" disabled={busy} onClick={() => void confirm(false)}>取消</button><button className="approve" disabled={busy} onClick={() => void confirm(true)}>确认执行</button></div></div>}
        <form onSubmit={(event) => { event.preventDefault(); void send(); }}>
          <textarea
            ref={inputRef}
            // 待确认期间服务端会以 409 拒绝新消息，与其让用户打完字再报错，不如直接说明。
            disabled={Boolean(result?.pending_confirmation)}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              // 输入法组合期间的回车是在确认候选词，不能当成发送——中文拼音下
              // 每选一次词都会把半截话发出去。
              if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
              event.preventDefault();
              void send();
            }}
            placeholder={result?.pending_confirmation ? "请先确认或取消上面的操作" : "描述你想办理的事情…（回车发送，Shift + 回车换行）"}
            rows={2}
          />
          <button disabled={busy || Boolean(result?.pending_confirmation)}>发送</button>
        </form>
      </div>
      <aside><div className="asideHead"><span>进行中</span>{matters.length > 0 && <small>{matters.length}</small>}</div>
        {/* 右栏只放有生命周期的事：还没办完的在这里，提交过的在下面的单据里。查询和
            闲聊办完就结束了，不占位置；执行过程在对话流里以步骤的形式出现。 */}
        {matters.length === 0 && <p className="asideIdle">当前没有进行中的事项</p>}
        <div className="matterList">{matters.map((matter) => <div className={`matter ${matter.status}`} key={matter.plan_id}>
          <div className="matterHead"><strong>{matter.title}</strong><em>{MATTER_STATUS_LABELS[matter.status]}</em></div>
          {(matter.known_fields.length > 0 || matter.missing_fields.length > 0) && <ul className="fields">
            {matter.known_fields.map((field) => <li key={field.name}><span>{field.label}</span><b>{field.value}</b>
              {/* 档案给的只是建议值，用户还没确认过，得和用户亲口说的区分开。 */}
              {field.source === "memory" && <i>建议</i>}</li>)}
            {matter.missing_fields.map((name) => <li key={name} className="missing"><span>{name}</span><b>待补充</b></li>)}
          </ul>}
          {matter.tasks.length > 1 && <div className="subtasks">{matter.tasks.filter((task) => task.id !== matter.task_id).map((task) =>
            <div key={task.id} className={task.status}><span>{task.status === "completed" ? "✓" : "○"}</span>{task.title}<em>{TASK_STATUS_LABELS[task.status] || task.status}</em></div>)}
          </div>}
          {/* 只往输入框里填一句话：字段仍由对话补充、由领域 Agent 解析，不开第二条提交路径。 */}
          {matter.status === "shelved" && <button className="resume" disabled={busy} onClick={() => {
            setInput(`继续办理「${matter.title}」`);
            inputRef.current?.focus();
          }}>继续</button>}
        </div>)}</div>
        <div className="asideHead actionsHead"><span>我的单据</span>{actions.length > 0 && <small>{actions.length}</small>}</div>
        {actions.length === 0 && <p className="asideIdle">这里会列出你提交过的单据</p>}
        <div className="actionList">{actions.map((item, index) => <div className="action" key={item.reference_id || `action-${index}`}>
          {/* 只分"已提交"和"已撤销"。workflow_actions 不知道外部系统的审批结果；
              不写状态，这份列表就会被整体读成"这些都批了"。 */}
          <div><strong>{ACTION_LABELS[item.action_type] || item.action_type}</strong><em>{item.revoked_at ? "已撤销" : "已提交"}</em><small>{item.created_at.slice(5, 10)}</small></div>
          {/* 字段名不翻译也不重排，顺序由后端白名单决定，前端只负责拼；枚举取值才翻译。 */}
          <p>{Object.values(item.fields).map((value) => VALUE_LABELS[value] ?? value).join(" · ")}</p>
          <code>{item.reference_id}</code>
        </div>)}</div>
      </aside>
    </section>
  </main>;
}

function App() {
  const [session, setSession] = useState<Session | null>(readSession);

  function signOut() {
    window.localStorage.removeItem(SESSION_KEY);
    setSession(null);
  }

  if (!session) return <LoginView onSignedIn={setSession}/>;
  // 换人登录要重建整棵子树，否则上一位的消息和任务面板会残留。
  return <ChatView key={session.userId} session={session} onSignOut={signOut}/>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App/></React.StrictMode>);

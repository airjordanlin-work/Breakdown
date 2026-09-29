import MoveLibrary from "./pages/MoveLibrary";
import Session from "./pages/Session";
import { useSession } from "./hooks/useSession";

export default function App() {
  const { sessionId, startSession, endSession } = useSession();
  return sessionId
    ? <Session sessionId={sessionId} onEnd={endSession} />
    : <MoveLibrary onStart={startSession} />;
}

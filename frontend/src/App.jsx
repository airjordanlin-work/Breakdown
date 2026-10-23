import MoveLibrary from "./pages/MoveLibrary";
import Session from "./pages/Session";
import Viewer from "./pages/Viewer";
import { useSession } from "./hooks/useSession";

// /view is the second-screen mirror for a TV or laptop: it follows the phone's session.
const isViewer = () => window.location.pathname.replace(/\/+$/, "") === "/view";

export default function App() {
  const { sessionId, startSession, endSession } = useSession();
  if (isViewer()) return <Viewer />;
  return sessionId
    ? <Session sessionId={sessionId} onEnd={endSession} />
    : <MoveLibrary onStart={startSession} />;
}

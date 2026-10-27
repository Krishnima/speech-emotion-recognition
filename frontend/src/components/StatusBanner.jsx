// Shown whenever the trained model isn't in use. The app still works (demo mode).
export default function StatusBanner({ status, onRetry }) {
  if (status.state === "ready") return null;

  if (status.state === "loading") {
    return <div className="banner">Connecting to the trained model. The first load can take a few seconds.</div>;
  }

  const retry = onRetry && (
    <button className="link" onClick={onRetry}>Check again</button>
  );

  let detail;
  if (status.state === "standalone") {
    detail = (
      <>This hosted version has no Python server, so results come from simple rules on pitch, loudness and pace.
        Run the project locally to get predictions from the trained CNN.</>
    );
  } else if (status.state === "offline") {
    detail = (
      <>The Python server isn’t running, so results come from simple rules on pitch, loudness and pace.
        Start it with <code>python app.py</code> to use the trained CNN. {retry}</>
    );
  } else if (status.state === "error") {
    detail = (
      <>The trained model couldn’t load (<code>{status.message}</code>). Make sure{" "}
        <code>pip install -r requirements.txt</code> finished, then restart <code>python app.py</code>. {retry}</>
    );
  } else {
    detail = (
      <>The CNN isn’t trained yet. Run <code>python train.py --data data/ravdess</code>, then restart{" "}
        <code>python app.py</code>. {retry}</>
    );
  }

  return (
    <div className={`banner${status.state === "error" ? " warn" : ""}`} data-testid="banner">
      <strong>Demo mode.</strong> {detail}
    </div>
  );
}

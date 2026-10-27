export default function Tabs({ tabs, active, onChange }) {
  function onKey(e) {
    const i = tabs.findIndex((t) => t.id === active);
    if (e.key === "ArrowRight") onChange(tabs[(i + 1) % tabs.length].id);
    if (e.key === "ArrowLeft") onChange(tabs[(i - 1 + tabs.length) % tabs.length].id);
  }
  return (
    <nav className="tabs" role="tablist" aria-label="Sections" onKeyDown={onKey}>
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          id={`tab-${t.id}`}
          aria-selected={active === t.id}
          aria-controls={`panel-${t.id}`}
          tabIndex={active === t.id ? 0 : -1}
          className={active === t.id ? "on" : ""}
          onClick={() => onChange(t.id)}
          data-testid={`tab-${t.id}`}
        >
          {t.label}
        </button>
      ))}
    </nav>
  );
}

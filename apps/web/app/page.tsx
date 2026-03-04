const phases = [
  {
    phase: "Phase 1",
    name: "Read Experience",
    scope: "Overview, corpus explorer, document library, and connector status",
    duration: "2 weeks"
  },
  {
    phase: "Phase 2",
    name: "Operator Workflows",
    scope: "Extraction controls, enrichment review queue, and policy delta workspace",
    duration: "2 weeks"
  },
  {
    phase: "Phase 3",
    name: "Cutover",
    scope: "Parallel run, acceptance checks, and Streamlit retirement",
    duration: "1-2 weeks"
  }
];

const launchTracks = [
  "API contracts finalized for metrics, documents, and jobs",
  "Background jobs wired to GitHub Actions with status endpoints",
  "Admin RBAC, audit logging, and observability baseline",
  "Performance budget and Lighthouse accessibility checks"
];

export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-6 px-4 py-6 md:px-8 md:py-10">
      <header className="panel reveal p-6 md:p-8">
        <span className="kicker">Migration Workspace</span>
        <div className="mt-4 grid gap-6 md:grid-cols-[1.3fr_1fr] md:items-end">
          <div>
            <h1 className="text-3xl font-bold leading-tight md:text-5xl">SEC Intelligence Console</h1>
            <p className="mt-3 max-w-2xl text-base text-[color:rgba(16,36,59,0.77)] md:text-lg">
              Streamlit is being replaced with a task-oriented Vercel app built for operators,
              analysts, and review workflows.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="panel p-4">
              <div className="metric">
                <span className="label">UI Platform</span>
                <span className="value">Next.js</span>
              </div>
            </div>
            <div className="panel p-4">
              <div className="metric">
                <span className="label">Host</span>
                <span className="value">Vercel</span>
              </div>
            </div>
            <div className="panel p-4">
              <div className="metric">
                <span className="label">Data Store</span>
                <span className="value">GCS JSON</span>
              </div>
            </div>
            <div className="panel p-4">
              <div className="metric">
                <span className="label">Workers</span>
                <span className="value">Python Jobs</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <section className="grid gap-4 md:grid-cols-3">
        {phases.map((item, index) => (
          <article
            key={item.phase}
            className={`panel reveal p-5 ${index === 1 ? "reveal-delay-1" : ""} ${index === 2 ? "reveal-delay-2" : ""}`}
          >
            <p className="text-sm font-semibold uppercase tracking-[0.13em] text-[color:rgba(45,86,115,0.8)]">{item.phase}</p>
            <h2 className="mt-2 text-2xl font-semibold">{item.name}</h2>
            <p className="mt-3 text-sm text-[color:rgba(16,36,59,0.78)]">{item.scope}</p>
            <p className="mt-4 inline-flex rounded-full border border-[color:rgba(16,36,59,0.18)] px-3 py-1 text-xs font-semibold tracking-[0.08em]">
              {item.duration}
            </p>
          </article>
        ))}
      </section>

      <section className="panel reveal reveal-delay-3 p-6 md:p-8">
        <h2 className="text-2xl font-semibold md:text-3xl">Launch Checklist</h2>
        <p className="mt-2 text-sm text-[color:rgba(16,36,59,0.75)]">
          This starter page is intentionally simple. Replace it with routed features as the API layer is delivered.
        </p>
        <div className="mt-5 grid gap-3 md:grid-cols-2">
          {launchTracks.map((item) => (
            <div key={item} className="rounded-xl border border-[color:var(--line)] bg-white/75 px-4 py-3 text-sm">
              {item}
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
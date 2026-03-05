function LineSkeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-[color:rgba(16,36,59,0.1)] ${className}`} />;
}

export default function Loading() {
  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 md:py-10">
      <section className="panel p-6 md:p-8">
        <LineSkeleton className="h-6 w-40" />
        <LineSkeleton className="mt-4 h-10 w-3/4" />
        <LineSkeleton className="mt-3 h-5 w-5/6" />
        <div className="mt-6 grid gap-3 md:grid-cols-4">
          {[1, 2, 3, 4].map((item) => (
            <div key={item} className="panel p-4">
              <LineSkeleton className="h-3 w-20" />
              <LineSkeleton className="mt-3 h-7 w-14" />
            </div>
          ))}
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-[1.6fr_1fr]">
        <article className="panel p-5">
          <LineSkeleton className="h-7 w-52" />
          <div className="mt-4 grid gap-2 sm:grid-cols-3">
            <LineSkeleton className="h-10 w-full" />
            <LineSkeleton className="h-10 w-full" />
            <LineSkeleton className="h-10 w-full" />
          </div>
          <div className="mt-4 space-y-2 rounded-xl border border-[color:var(--line)] bg-white p-3">
            {[1, 2, 3, 4, 5].map((row) => (
              <LineSkeleton key={row} className="h-8 w-full" />
            ))}
          </div>
        </article>

        <article className="panel p-5">
          <LineSkeleton className="h-7 w-48" />
          <LineSkeleton className="mt-3 h-4 w-5/6" />
          <div className="mt-4 space-y-3">
            <LineSkeleton className="h-40 w-full" />
            <LineSkeleton className="h-48 w-full" />
            <LineSkeleton className="h-24 w-full" />
          </div>
        </article>
      </section>
    </div>
  );
}
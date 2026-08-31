import { type NextRequest, NextResponse } from "next/server";
import { generateCandidateArticle } from "@/lib/server/scheduled-editorial";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = await req.json() as {
      runId?: unknown;
      outputId?: unknown;
      candidateId?: unknown;
      regenerate?: unknown;
    };
    const draft = await generateCandidateArticle({
      runId: Number(body.runId),
      outputId: Number(body.outputId),
      candidateId: String(body.candidateId || ""),
      regenerate: body.regenerate === true,
    });
    return NextResponse.json({ ok: true, data: { draft } });
  } catch (error) {
    console.error("[admin/briefings/scheduled/candidate-draft POST]", error);
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : String(error) },
      { status: 500 },
    );
  }
}

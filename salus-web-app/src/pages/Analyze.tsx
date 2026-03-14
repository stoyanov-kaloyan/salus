import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import SiteHeader from "@/components/SiteHeader";
import {
    api,
    type DetectionPrediction,
    type MultiDetectionResult,
} from "@/lib/api";

const RECOGNIZER_ORDER = ["deepfake", "nsfw", "flux"];

const RECOGNIZER_LABEL: Record<string, string> = {
    deepfake: "Deepfake",
    nsfw: "NSFW",
    flux: "Flux",
};

const PREDICTION_LABEL_KEYS = ["label", "class", "name", "category"];
const PREDICTION_SCORE_KEYS = ["score", "confidence", "probability", "value"];

function formatPercent(score: number): string {
    const percent = score <= 1 ? score * 100 : score;
    return `${percent.toFixed(1)}%`;
}

function getPredictionLabel(prediction: DetectionPrediction): string {
    for (const key of PREDICTION_LABEL_KEYS) {
        const value = prediction[key];
        if (typeof value === "string" && value.trim().length > 0) {
            return value;
        }
    }
    return "Unknown";
}

function getPredictionScore(prediction: DetectionPrediction): number | null {
    for (const key of PREDICTION_SCORE_KEYS) {
        const value = prediction[key];
        if (typeof value === "number") {
            return value;
        }
        if (typeof value === "string") {
            const parsed = Number(value);
            if (!Number.isNaN(parsed)) {
                return parsed;
            }
        }
    }
    return null;
}

const Analyze = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [analysis, setAnalysis] = useState<MultiDetectionResult | null>(null);
    const [isRunning, setIsRunning] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!selectedFile) {
            setPreviewUrl(null);
            return;
        }

        const url = URL.createObjectURL(selectedFile);
        setPreviewUrl(url);

        return () => {
            URL.revokeObjectURL(url);
        };
    }, [selectedFile]);

    const detectorResults = useMemo(() => {
        if (!analysis) return [];

        return Object.entries(analysis.results).sort(([a], [b]) => {
            const rankA = RECOGNIZER_ORDER.indexOf(a);
            const rankB = RECOGNIZER_ORDER.indexOf(b);
            const normalizedA = rankA === -1 ? 999 : rankA;
            const normalizedB = rankB === -1 ? 999 : rankB;
            return normalizedA - normalizedB;
        });
    }, [analysis]);

    const flaggedDetectors = useMemo(
        () => detectorResults.filter(([, result]) => result.is_target),
        [detectorResults],
    );

    const averageScore = useMemo(() => {
        if (!detectorResults.length) return null;
        const total = detectorResults.reduce((sum, [, result]) => {
            return sum + result.score;
        }, 0);
        return total / detectorResults.length;
    }, [detectorResults]);

    async function runAnalysis(event: FormEvent<HTMLFormElement>) {
        event.preventDefault();

        if (!selectedFile) {
            setError("Please select an image first.");
            return;
        }

        setIsRunning(true);
        setError(null);

        try {
            const nextAnalysis = await api.detectImageMulti(selectedFile);
            setAnalysis(nextAnalysis);
        } catch (caughtError) {
            const message =
                caughtError instanceof Error
                    ? caughtError.message
                    : "Unable to run detection right now.";
            setError(message);
            setAnalysis(null);
        } finally {
            setIsRunning(false);
        }
    }

    return (
        <div className="min-h-screen bg-background text-foreground">
            <SiteHeader active="analyze" />

            <main className="container max-w-6xl mx-auto px-6 py-12">
                <section className="border p-6 md:p-8">
                    <h1 className="font-serif text-4xl md:text-5xl tracking-tight">
                        Try Live Image Detection
                    </h1>
                    <p className="mt-4 text-muted-foreground max-w-3xl leading-relaxed">
                        Upload one image and we will run a full multi-detector
                        pass using our Deepfake, NSFW, and Flux recognizers.
                    </p>

                    <form
                        onSubmit={runAnalysis}
                        className="mt-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
                        <div className="lg:col-span-5">
                            <div className="aspect-[4/3] border border-dashed border-foreground/40 p-2 flex items-center justify-center bg-muted/20 overflow-hidden">
                                {previewUrl ? (
                                    <img
                                        src={previewUrl}
                                        alt="Selected upload preview"
                                        className="w-full h-full object-contain"
                                    />
                                ) : (
                                    <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground text-center">
                                        Image preview appears here
                                    </p>
                                )}
                            </div>
                        </div>

                        <div className="lg:col-span-7 space-y-5">
                            <div>
                                <label
                                    htmlFor="analysis-file"
                                    className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
                                    Upload Image
                                </label>
                                <input
                                    id="analysis-file"
                                    type="file"
                                    accept="image/*"
                                    onChange={(event) => {
                                        const nextFile =
                                            event.target.files?.[0] ?? null;
                                        setSelectedFile(nextFile);
                                        setAnalysis(null);
                                        setError(null);
                                    }}
                                    className="mt-2 block w-full text-sm file:mr-4 file:py-2 file:px-4 file:border file:border-foreground file:bg-background file:font-mono file:text-xs file:uppercase file:tracking-wider"
                                />
                            </div>

                            <div className="border p-4">
                                <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground mb-2">
                                    Detector Set
                                </p>
                                <p className="text-sm leading-relaxed">
                                    Deepfake, NSFW, and Flux
                                </p>
                            </div>

                            <button
                                type="submit"
                                disabled={!selectedFile || isRunning}
                                className="inline-flex items-center gap-2 bg-primary text-primary-foreground font-mono text-sm uppercase tracking-wider px-6 py-3 border border-foreground shadow-[3px_3px_0px_0px_hsl(var(--foreground))] hover:shadow-none hover:translate-x-[3px] hover:translate-y-[3px] transition-all disabled:opacity-50 disabled:pointer-events-none">
                                {isRunning
                                    ? "Analyzing..."
                                    : "Run Full Analysis"}
                            </button>

                            {selectedFile ? (
                                <p className="font-mono text-xs text-muted-foreground">
                                    Selected: {selectedFile.name}
                                </p>
                            ) : null}
                        </div>
                    </form>

                    {error ? (
                        <div className="mt-6 border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                            {error}
                        </div>
                    ) : null}
                </section>

                {analysis ? (
                    <section className="mt-10 border p-6 md:p-8">
                        <h2 className="font-serif text-3xl md:text-4xl tracking-tight">
                            Full Analysis
                        </h2>

                        <div className="mt-6 grid grid-cols-1 md:grid-cols-4 divide-y md:divide-y-0 md:divide-x border">
                            <Metric
                                value={
                                    flaggedDetectors.length > 0
                                        ? "FLAGGED"
                                        : "CLEAN"
                                }
                                label="Overall Verdict"
                            />
                            <Metric
                                value={String(detectorResults.length)}
                                label="Detectors Run"
                            />
                            <Metric
                                value={String(flaggedDetectors.length)}
                                label="Flagged Detectors"
                            />
                            <Metric
                                value={
                                    averageScore == null
                                        ? "—"
                                        : formatPercent(averageScore)
                                }
                                label="Average Confidence"
                            />
                        </div>

                        <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
                            {detectorResults.map(([name, result]) => {
                                const sortedPredictions = [
                                    ...result.all_predictions,
                                ].sort((a, b) => {
                                    const scoreA = getPredictionScore(a);
                                    const scoreB = getPredictionScore(b);
                                    return (scoreB ?? -1) - (scoreA ?? -1);
                                });

                                return (
                                    <article key={name} className="border p-5">
                                        <div className="flex items-center justify-between gap-3">
                                            <h3 className="font-serif text-2xl">
                                                {RECOGNIZER_LABEL[name] ??
                                                    name.toUpperCase()}
                                            </h3>
                                            <span
                                                className={`font-mono text-[10px] uppercase tracking-wider px-2 py-1 border ${
                                                    result.is_target
                                                        ? "bg-destructive/10 border-destructive"
                                                        : "bg-primary/20 border-primary"
                                                }`}>
                                                {result.is_target
                                                    ? "Flagged"
                                                    : "Clean"}
                                            </span>
                                        </div>

                                        <div className="mt-4 space-y-2 text-sm">
                                            <p>
                                                <span className="font-mono text-xs uppercase tracking-wider text-muted-foreground mr-2">
                                                    Label
                                                </span>
                                                {result.label}
                                            </p>
                                            <p>
                                                <span className="font-mono text-xs uppercase tracking-wider text-muted-foreground mr-2">
                                                    Confidence
                                                </span>
                                                {formatPercent(result.score)}
                                            </p>
                                        </div>

                                        <div className="mt-5">
                                            <h4 className="font-mono text-xs uppercase tracking-wider text-muted-foreground mb-3">
                                                Prediction Breakdown
                                            </h4>

                                            {sortedPredictions.length === 0 ? (
                                                <p className="text-sm text-muted-foreground">
                                                    No prediction details were
                                                    returned.
                                                </p>
                                            ) : (
                                                <ul className="space-y-2">
                                                    {sortedPredictions
                                                        .slice(0, 6)
                                                        .map(
                                                            (prediction, i) => {
                                                                const label =
                                                                    getPredictionLabel(
                                                                        prediction,
                                                                    );
                                                                const score =
                                                                    getPredictionScore(
                                                                        prediction,
                                                                    );

                                                                return (
                                                                    <li
                                                                        key={`${name}-pred-${i}`}
                                                                        className="flex items-center justify-between gap-3 border-b border-border pb-2">
                                                                        <span className="text-sm truncate">
                                                                            {
                                                                                label
                                                                            }
                                                                        </span>
                                                                        <span className="font-mono text-xs text-muted-foreground whitespace-nowrap">
                                                                            {score ==
                                                                            null
                                                                                ? "n/a"
                                                                                : formatPercent(
                                                                                      score,
                                                                                  )}
                                                                        </span>
                                                                    </li>
                                                                );
                                                            },
                                                        )}
                                                </ul>
                                            )}
                                        </div>
                                    </article>
                                );
                            })}
                        </div>
                    </section>
                ) : null}
            </main>

            <footer className="py-6 border-t mt-8">
                <div className="container max-w-6xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
                    <span className="font-mono text-xs text-muted-foreground">
                        © {new Date().getFullYear()} Salus
                    </span>
                    <span className="font-mono text-xs text-muted-foreground">
                        Multi-Detector Testing Page
                    </span>
                </div>
            </footer>
        </div>
    );
};

const Metric = ({ value, label }: { value: string; label: string }) => (
    <div className="p-4 md:p-5 text-center md:text-left">
        <span className="block font-mono text-2xl md:text-3xl font-bold tracking-tight">
            {value}
        </span>
        <span className="block font-mono text-[10px] uppercase tracking-wider text-muted-foreground mt-1">
            {label}
        </span>
    </div>
);

export default Analyze;

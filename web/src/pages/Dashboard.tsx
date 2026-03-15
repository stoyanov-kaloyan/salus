import { useMemo } from "react";
import { parseISO, formatDistanceToNow } from "date-fns";
import { useSummary, useEvents } from "@/hooks/useStats";
import SiteHeader from "@/components/SiteHeader";
import {
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart";
import {
    Bar,
    BarChart,
    XAxis,
    YAxis,
    CartesianGrid,
    Pie,
    PieChart,
    Cell,
} from "recharts";

const PIE_COLORS = ["hsl(205, 86%, 76%)", "hsl(40, 10%, 90%)"];

const chartConfig = {
    threats: { label: "Threats", color: "hsl(205, 86%, 76%)" },
    count: { label: "Count", color: "hsl(205, 86%, 76%)" },
};

const Dashboard = () => {
    const { data: summary } = useSummary();
    const { data: recentEvents } = useEvents({ flagged_only: true, limit: 8 });

    const totalScans = useMemo(
        () => (summary ?? []).reduce((s, r) => s + r.total_scans, 0),
        [summary],
    );
    const totalFlagged = useMemo(
        () => (summary ?? []).reduce((s, r) => s + r.flagged_count, 0),
        [summary],
    );
    const blockRate =
        totalScans > 0
            ? `${((totalFlagged / totalScans) * 100).toFixed(1)}%`
            : "—";
    const avgConfidence = useMemo(() => {
        const valid = (summary ?? []).filter(
            (r) => r.avg_flagged_score != null,
        );
        if (!valid.length) return "—";
        const avg =
            valid.reduce((s, r) => s + (r.avg_flagged_score ?? 0), 0) /
            valid.length;
        return `${(avg * 100).toFixed(1)}%`;
    }, [summary]);
    const activeFilters = (summary ?? []).length;
    const lastUpdated = useMemo(() => {
        const dates = (summary ?? [])
            .filter((r) => r.last_scan_at)
            .map((r) => r.last_scan_at!);
        if (!dates.length) return "just now";
        return formatDistanceToNow(parseISO(dates.sort().at(-1)!), {
            addSuffix: true,
        });
    }, [summary]);

    const categoryData = useMemo(
        () =>
            (summary ?? []).map((r) => ({
                category:
                    r.recognizer.charAt(0).toUpperCase() +
                    r.recognizer.slice(1),
                count: r.total_scans,
            })),
        [summary],
    );

    const pieData = useMemo(() => {
        if (!summary?.length) return [];
        return [
            { name: "Flagged", value: totalFlagged },
            { name: "Passed", value: totalScans - totalFlagged },
        ];
    }, [summary, totalFlagged, totalScans]);

    const threatFeed = useMemo(
        () =>
            (recentEvents ?? []).map((e, i) => ({
                id: i,
                type:
                    e.recognizer.charAt(0).toUpperCase() +
                    e.recognizer.slice(1),
                domain: e.image_hash
                    ? `sha256:${e.image_hash.slice(0, 16)}`
                    : e.endpoint,
                time: formatDistanceToNow(parseISO(e.created_at), {
                    addSuffix: true,
                }),
                confidence: Math.round(e.score * 1000) / 10,
            })),
        [recentEvents],
    );

    return (
        <div className="min-h-screen bg-background text-foreground">
            <SiteHeader active="dashboard" />

            <main className="container max-w-7xl mx-auto px-6 py-8">
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-0 divide-y lg:divide-y-0 lg:divide-x divide-border">
                    <div className="lg:col-span-3 lg:pr-8 pb-8 lg:pb-0">
                        <h2 className="font-mono text-xs uppercase tracking-wider text-muted-foreground mb-6">
                            Today's Figures
                        </h2>
                        <div className="space-y-6">
                            <StatBlock
                                value={totalFlagged.toLocaleString()}
                                label="Threats Detected"
                            />
                            <StatBlock
                                value={avgConfidence}
                                label="Avg. Confidence"
                            />
                            <StatBlock
                                value={
                                    activeFilters ? String(activeFilters) : "—"
                                }
                                label="Active Recognizers"
                            />
                            <StatBlock
                                value={totalScans.toLocaleString()}
                                label="Total Scans"
                            />
                            <StatBlock value={blockRate} label="Flag Rate" />
                        </div>
                    </div>

                    <div className="lg:col-span-5 lg:px-8 py-8 lg:py-0">
                        <h2 className="font-mono text-xs uppercase tracking-wider text-muted-foreground mb-6">
                            Live Threat Feed
                        </h2>
                        <div className="space-y-0 divide-y divide-border">
                            {threatFeed.length === 0 ? (
                                <p className="font-mono text-xs text-muted-foreground">
                                    No flagged events yet.
                                </p>
                            ) : (
                                threatFeed.map((t) => (
                                    <div key={t.id} className="py-4 first:pt-0">
                                        <div className="flex items-start justify-between gap-3">
                                            <div className="flex-1 min-w-0">
                                                <div className="flex items-center gap-2 mb-1">
                                                    <span
                                                        className={`font-mono text-[10px] uppercase tracking-wider px-1.5 py-0.5 border ${
                                                            t.type === "NSFW"
                                                                ? "bg-primary/20 border-primary"
                                                                : t.type ===
                                                                    "Malicious"
                                                                  ? "bg-destructive/10 border-destructive"
                                                                  : "border-foreground"
                                                        }`}>
                                                        {t.type}
                                                    </span>
                                                    <span className="font-mono text-[10px] text-muted-foreground">
                                                        {t.time}
                                                    </span>
                                                </div>
                                                <p className="font-serif text-lg leading-tight truncate">
                                                    {t.domain}
                                                </p>
                                            </div>
                                            <span className="font-mono text-xs text-muted-foreground whitespace-nowrap mt-1">
                                                {t.confidence}%
                                            </span>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    <div className="lg:col-span-4 lg:pl-8 pt-8 lg:pt-0">
                        <h2 className="font-mono text-xs uppercase tracking-wider text-muted-foreground mb-6">
                            Analytics
                        </h2>
                        {/* Bar Chart — Categories */}
                        <div className="mb-8">
                            <h3 className="font-serif text-xl mb-4">
                                By Category
                            </h3>
                            <div className="border p-4">
                                <ChartContainer
                                    config={chartConfig}
                                    className="h-[180px] w-full">
                                    <BarChart
                                        data={categoryData}
                                        margin={{
                                            top: 5,
                                            right: 5,
                                            bottom: 5,
                                            left: -20,
                                        }}>
                                        <CartesianGrid
                                            strokeDasharray="3 3"
                                            stroke="hsl(220, 20%, 12%)"
                                            opacity={0.15}
                                        />
                                        <XAxis
                                            dataKey="category"
                                            tick={{
                                                fontSize: 11,
                                                fontFamily: "'JetBrains Mono'",
                                            }}
                                            stroke="hsl(220, 20%, 12%)"
                                            strokeWidth={0.5}
                                        />
                                        <YAxis
                                            tick={{
                                                fontSize: 11,
                                                fontFamily: "'JetBrains Mono'",
                                            }}
                                            stroke="hsl(220, 20%, 12%)"
                                            strokeWidth={0.5}
                                        />
                                        <ChartTooltip
                                            content={<ChartTooltipContent />}
                                        />
                                        <Bar
                                            dataKey="count"
                                            fill="hsl(205, 86%, 76%)"
                                            stroke="hsl(220, 20%, 12%)"
                                            strokeWidth={1}
                                        />
                                    </BarChart>
                                </ChartContainer>
                            </div>
                        </div>

                        {/* Pie Chart — Block Rate */}
                        <div>
                            <h3 className="font-serif text-xl mb-4">
                                Traffic Breakdown
                            </h3>
                            <div className="border p-4">
                                <ChartContainer
                                    config={chartConfig}
                                    className="h-[200px] w-full">
                                    <PieChart>
                                        <Pie
                                            data={pieData}
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={50}
                                            outerRadius={80}
                                            dataKey="value"
                                            stroke="hsl(220, 20%, 12%)"
                                            strokeWidth={1}>
                                            {pieData.map((_, index) => (
                                                <Cell
                                                    key={index}
                                                    fill={PIE_COLORS[index]}
                                                />
                                            ))}
                                        </Pie>
                                        <ChartTooltip
                                            content={<ChartTooltipContent />}
                                        />
                                    </PieChart>
                                </ChartContainer>
                                <div className="flex justify-center gap-4 mt-3">
                                    {pieData.map((entry, i) => (
                                        <div
                                            key={entry.name}
                                            className="flex items-center gap-1.5">
                                            <div
                                                className="w-2.5 h-2.5 border border-foreground"
                                                style={{
                                                    backgroundColor:
                                                        PIE_COLORS[i],
                                                }}
                                            />
                                            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                                                {entry.name}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </main>

            <footer className="border-t py-6 mt-8">
                <div className="container max-w-7xl mx-auto px-6 flex items-center justify-between">
                    <span className="font-mono text-xs text-muted-foreground">
                        © {new Date().getFullYear()} Salus
                    </span>
                    <span className="font-mono text-xs text-muted-foreground">
                        Last updated: {lastUpdated}
                    </span>
                </div>
            </footer>
        </div>
    );
};

const StatBlock = ({ value, label }: { value: string; label: string }) => (
    <div className="pb-4 border-b border-primary/40">
        <span className="block font-mono text-3xl font-bold tracking-tight">
            {value}
        </span>
        <span className="block font-mono text-[10px] uppercase tracking-wider text-muted-foreground mt-1">
            {label}
        </span>
    </div>
);

export default Dashboard;

import { Link } from "react-router-dom";
import { Separator } from "@/components/ui/separator";
import {
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart";
import {
    Line,
    LineChart,
    Bar,
    BarChart,
    XAxis,
    YAxis,
    CartesianGrid,
    Pie,
    PieChart,
    Cell,
} from "recharts";

const today = new Date().toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
});

// Mock data
const weeklyData = [
    { day: "Mon", threats: 142 },
    { day: "Tue", threats: 189 },
    { day: "Wed", threats: 164 },
    { day: "Thu", threats: 221 },
    { day: "Fri", threats: 198 },
    { day: "Sat", threats: 87 },
    { day: "Sun", threats: 63 },
];

const categoryData = [
    { category: "NSFW", count: 487 },
    { category: "Deepfake", count: 156 },
    { category: "Malicious", count: 321 },
];

const pieData = [
    { name: "Blocked", value: 964 },
    { name: "Flagged", value: 89 },
    { name: "Passed", value: 12847 },
];

const PIE_COLORS = [
    "hsl(205, 86%, 76%)",
    "hsl(220, 20%, 12%)",
    "hsl(40, 10%, 90%)",
];

const threatFeed = [
    {
        id: 1,
        type: "NSFW",
        domain: "cdn.suspect-img.net",
        time: "2 min ago",
        confidence: 98.4,
    },
    {
        id: 2,
        type: "Malicious",
        domain: "login-verify.phishbank.ru",
        time: "5 min ago",
        confidence: 99.1,
    },
    {
        id: 3,
        type: "Deepfake",
        domain: "media.synth-face.io",
        time: "8 min ago",
        confidence: 94.7,
    },
    {
        id: 4,
        type: "NSFW",
        domain: "img-host.adultcdn.com",
        time: "12 min ago",
        confidence: 97.2,
    },
    {
        id: 5,
        type: "Malicious",
        domain: "update.win-security.xyz",
        time: "15 min ago",
        confidence: 99.8,
    },
    {
        id: 6,
        type: "Deepfake",
        domain: "video.ai-clone.net",
        time: "21 min ago",
        confidence: 91.3,
    },
    {
        id: 7,
        type: "Malicious",
        domain: "api.crypto-drain.io",
        time: "28 min ago",
        confidence: 98.9,
    },
    {
        id: 8,
        type: "NSFW",
        domain: "stream.explicit-live.com",
        time: "34 min ago",
        confidence: 96.1,
    },
];

const chartConfig = {
    threats: {
        label: "Threats",
        color: "hsl(205, 86%, 76%)",
    },
    count: {
        label: "Count",
        color: "hsl(205, 86%, 76%)",
    },
};

const Dashboard = () => {
    return (
        <div className="min-h-screen bg-background text-foreground">
            {/* Masthead */}
            <header className="border-b">
                <div className="container max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <span className="font-mono text-xs text-muted-foreground uppercase tracking-wider">
                            {today}
                        </span>
                        <Link
                            to="/"
                            className="text-4xl md:text-5xl font-serif tracking-tight hover:text-accent transition-colors">
                            SALUS
                        </Link>
                        <span className="font-mono text-xs text-muted-foreground uppercase tracking-wider">
                            The Ledger
                        </span>
                    </div>
                </div>
                <Separator />
                <div className="container max-w-7xl mx-auto px-6 py-2 flex justify-center gap-8">
                    <Link
                        to="/"
                        className="font-mono text-xs uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors">
                        Front Page
                    </Link>
                    <Link
                        to="/dashboard"
                        className="font-mono text-xs uppercase tracking-wider text-foreground">
                        The Ledger
                    </Link>
                </div>
            </header>

            {/* Dashboard Content */}
            <main className="container max-w-7xl mx-auto px-6 py-8">
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-0 divide-y lg:divide-y-0 lg:divide-x divide-border">
                    {/* Left Column — Stats */}
                    <div className="lg:col-span-3 lg:pr-8 pb-8 lg:pb-0">
                        <h2 className="font-mono text-xs uppercase tracking-wider text-muted-foreground mb-6">
                            Today's Figures
                        </h2>
                        <div className="space-y-6">
                            <StatBlock value="964" label="Threats Blocked" />
                            <StatBlock
                                value="99.2%"
                                label="Detection Accuracy"
                            />
                            <StatBlock value="3" label="Active Filters" />
                            <StatBlock value="14ms" label="Avg. Response" />
                            <StatBlock value="13,900" label="Total Requests" />
                            <StatBlock value="6.9%" label="Block Rate" />
                        </div>
                    </div>

                    {/* Center Column — Live Feed */}
                    <div className="lg:col-span-5 lg:px-8 py-8 lg:py-0">
                        <h2 className="font-mono text-xs uppercase tracking-wider text-muted-foreground mb-6">
                            Live Threat Feed
                        </h2>
                        <div className="space-y-0 divide-y divide-border">
                            {threatFeed.map((t) => (
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
                            ))}
                        </div>
                    </div>

                    {/* Right Column — Charts */}
                    <div className="lg:col-span-4 lg:pl-8 pt-8 lg:pt-0">
                        <h2 className="font-mono text-xs uppercase tracking-wider text-muted-foreground mb-6">
                            Analytics
                        </h2>

                        {/* Line Chart — Weekly Threats */}
                        <div className="mb-8">
                            <h3 className="font-serif text-xl mb-4">
                                7-Day Trend
                            </h3>
                            <div className="border p-4">
                                <ChartContainer
                                    config={chartConfig}
                                    className="h-[180px] w-full">
                                    <LineChart
                                        data={weeklyData}
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
                                            dataKey="day"
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
                                        <Line
                                            type="linear"
                                            dataKey="threats"
                                            stroke="hsl(205, 86%, 76%)"
                                            strokeWidth={2}
                                            dot={{
                                                r: 3,
                                                fill: "hsl(205, 86%, 76%)",
                                                stroke: "hsl(220, 20%, 12%)",
                                                strokeWidth: 1,
                                            }}
                                        />
                                    </LineChart>
                                </ChartContainer>
                            </div>
                        </div>

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

            {/* Footer */}
            <footer className="border-t py-6 mt-8">
                <div className="container max-w-7xl mx-auto px-6 flex items-center justify-between">
                    <span className="font-mono text-xs text-muted-foreground">
                        © {new Date().getFullYear()} Salus
                    </span>
                    <span className="font-mono text-xs text-muted-foreground">
                        Last updated: just now
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

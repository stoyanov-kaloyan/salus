import { Link } from "react-router-dom";
import { Shield, Eye, Globe, ArrowRight } from "lucide-react";
import { Separator } from "@/components/ui/separator";

const today = new Date().toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
});

const features = [
    {
        icon: Eye,
        title: "NSFW Detection",
        desc: "Real-time classification of explicit imagery and text using multi-modal AI. Content is intercepted before it reaches the end user.",
    },
    {
        icon: Shield,
        title: "Deepfake Identification",
        desc: "Facial manipulation and synthetic media detection across video, image, and audio streams with frame-level analysis.",
    },
    {
        icon: Globe,
        title: "Malicious Domain Blocking",
        desc: "Continuous threat intelligence feed cross-referenced against live DNS queries. Phishing, malware, and C2 domains neutralized.",
    },
];

const stats = [
    { value: "99.2%", label: "Accuracy" },
    { value: "14ms", label: "Avg. Latency" },
    { value: "1M+", label: "Threats Blocked" },
];

const steps = [
    {
        num: "01",
        title: "Route Traffic",
        desc: "Point your network's DNS or proxy configuration to Salus. Setup takes under five minutes.",
    },
    {
        num: "02",
        title: "AI Inspects",
        desc: "Every request passes through our inference pipeline — images, text, URLs — all classified in real time.",
    },
    {
        num: "03",
        title: "Threats Blocked",
        desc: "Harmful content never reaches the user. Clean traffic flows through. Every incident is logged for review.",
    },
];

const Index = () => {
    return (
        <div className="min-h-screen bg-background text-foreground">
            {/* Masthead */}
            <header className="border-b">
                <div className="container max-w-6xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <span className="font-mono text-xs text-muted-foreground uppercase tracking-wider">
                            {today}
                        </span>
                        <h1 className="text-5xl md:text-6xl font-serif tracking-tight">
                            SALUS
                        </h1>
                        <span className="font-mono text-xs text-muted-foreground uppercase tracking-wider">
                            Status:{" "}
                            <span className="text-foreground">Secure</span>
                        </span>
                    </div>
                </div>
                <Separator />
                <div className="container max-w-6xl mx-auto px-6 py-2 flex justify-center gap-8">
                    <Link
                        to="/"
                        className="font-mono text-xs uppercase tracking-wider text-foreground hover:text-accent transition-colors">
                        Front Page
                    </Link>
                    <Link
                        to="/dashboard"
                        className="font-mono text-xs uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors">
                        The Ledger
                    </Link>
                </div>
            </header>

            {/* Hero */}
            <section className="border-b">
                <div className="container max-w-6xl mx-auto px-6 py-16 md:py-24">
                    <h2 className="text-4xl md:text-6xl lg:text-7xl font-serif leading-[1.1] max-w-4xl">
                        Your Network's
                        <br />
                        Daily Guardian.
                    </h2>
                    <p className="mt-6 text-lg md:text-xl text-muted-foreground max-w-2xl leading-relaxed">
                        Salus is an AI-powered content filter and proxy that
                        stands between your users and the worst of the internet
                        — blocking NSFW material, synthetic media, and malicious
                        domains before they cause harm.
                    </p>
                    <div className="mt-10 flex flex-wrap gap-4">
                        <Link
                            to="/dashboard"
                            className="inline-flex items-center gap-2 bg-primary text-primary-foreground font-mono text-sm uppercase tracking-wider px-6 py-3 border border-foreground shadow-[3px_3px_0px_0px_hsl(var(--foreground))] hover:shadow-none hover:translate-x-[3px] hover:translate-y-[3px] transition-all">
                            Open the Ledger
                            <ArrowRight className="w-4 h-4" />
                        </Link>
                        <a
                            href="#how-it-works"
                            className="inline-flex items-center gap-2 bg-background text-foreground font-mono text-sm uppercase tracking-wider px-6 py-3 border border-foreground hover:bg-foreground hover:text-background transition-colors">
                            How It Works
                        </a>
                    </div>
                </div>
            </section>

            {/* Features */}
            <section className="border-b">
                <div className="container max-w-6xl mx-auto px-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-border">
                        {features.map((f) => (
                            <div
                                key={f.title}
                                className="py-10 md:px-8 first:md:pl-0 last:md:pr-0">
                                <f.icon
                                    className="w-6 h-6 text-accent mb-4"
                                    strokeWidth={1.5}
                                />
                                <h3 className="text-2xl font-serif mb-3">
                                    {f.title}
                                </h3>
                                <p className="text-sm text-muted-foreground leading-relaxed">
                                    {f.desc}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Stats */}
            <section className="border-b">
                <div className="container max-w-6xl mx-auto px-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-border">
                        {stats.map((s) => (
                            <div
                                key={s.label}
                                className="py-10 md:px-8 first:md:pl-0 last:md:pr-0 text-center md:text-left">
                                <span className="block font-mono text-4xl md:text-5xl font-bold tracking-tight">
                                    {s.value}
                                </span>
                                <span className="block font-mono text-xs text-muted-foreground uppercase tracking-wider mt-2">
                                    {s.label}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* How It Works */}
            <section id="how-it-works" className="border-b">
                <div className="container max-w-6xl mx-auto px-6 py-16">
                    <h2 className="text-3xl md:text-4xl font-serif mb-12">
                        How It Works
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-0 divide-y md:divide-y-0 md:divide-x divide-border">
                        {steps.map((s) => (
                            <div
                                key={s.num}
                                className="py-8 md:px-8 first:md:pl-0 last:md:pr-0">
                                <span className="font-mono text-xs text-accent uppercase tracking-wider">
                                    Step {s.num}
                                </span>
                                <h3 className="text-2xl font-serif mt-2 mb-3">
                                    {s.title}
                                </h3>
                                <p className="text-sm text-muted-foreground leading-relaxed">
                                    {s.desc}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className="border-b">
                <div className="container max-w-6xl mx-auto px-6 py-16 text-center">
                    <h2 className="text-3xl md:text-4xl font-serif mb-4">
                        Start Protecting Your Network
                    </h2>
                    <p className="text-muted-foreground mb-8 max-w-lg mx-auto">
                        Deploy Salus in minutes. No hardware required.
                        AI-powered protection from day one.
                    </p>
                    <Link
                        to="/dashboard"
                        className="inline-flex items-center gap-2 bg-primary text-primary-foreground font-mono text-sm uppercase tracking-wider px-8 py-4 border border-foreground shadow-[4px_4px_0px_0px_hsl(var(--foreground))] hover:shadow-none hover:translate-x-[4px] hover:translate-y-[4px] transition-all">
                        Open Dashboard
                        <ArrowRight className="w-4 h-4" />
                    </Link>
                </div>
            </section>

            {/* Footer */}
            <footer className="py-6">
                <div className="container max-w-6xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
                    <span className="font-mono text-xs text-muted-foreground">
                        © {new Date().getFullYear()} Salus. All rights reserved.
                    </span>
                    <span className="font-mono text-xs text-muted-foreground">
                        AI Content Filter & Proxy
                    </span>
                </div>
            </footer>
        </div>
    );
};

export default Index;

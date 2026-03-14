import { Link } from "react-router-dom";
import { Separator } from "@/components/ui/separator";

type SiteHeaderProps = {
    active: "home" | "dashboard";
};

const today = new Date().toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
});

const activeLinkClass =
    "font-mono text-xs uppercase tracking-wider text-foreground hover:text-accent transition-colors";
const inactiveLinkClass =
    "font-mono text-xs uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors";

const SiteHeader = ({ active }: SiteHeaderProps) => {
    return (
        <header className="border-b">
            <div className="container max-w-6xl mx-auto px-6 py-4">
                <div className="flex items-center justify-between">
                    <span className="font-mono text-xs text-muted-foreground uppercase tracking-wider">
                        {today}
                    </span>
                    <Link
                        to="/"
                        className="text-5xl md:text-6xl font-serif tracking-tight hover:text-accent transition-colors">
                        SALUS
                    </Link>
                    <span className="font-mono text-xs text-muted-foreground uppercase tracking-wider">
                        Status: <span className="text-foreground">Secure</span>
                    </span>
                </div>
            </div>
            <Separator />
            <div className="container max-w-6xl mx-auto px-6 py-2 flex justify-center gap-8">
                <Link
                    to="/"
                    className={
                        active === "home" ? activeLinkClass : inactiveLinkClass
                    }>
                    Front Page
                </Link>
                <Link
                    to="/dashboard"
                    className={
                        active === "dashboard"
                            ? activeLinkClass
                            : inactiveLinkClass
                    }>
                    The Ledger
                </Link>
            </div>
        </header>
    );
};

export default SiteHeader;

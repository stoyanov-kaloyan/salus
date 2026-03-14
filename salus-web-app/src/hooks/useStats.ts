import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useSummary() {
    return useQuery({
        queryKey: ["stats", "summary"],
        queryFn: api.getSummary,
        refetchInterval: 30_000,
    });
}

export function useDaily(params?: { recognizer?: string; days?: number }) {
    return useQuery({
        queryKey: ["stats", "daily", params],
        queryFn: () => api.getDaily(params),
        refetchInterval: 60_000,
    });
}

export function useRecentActivity() {
    return useQuery({
        queryKey: ["stats", "recent"],
        queryFn: api.getRecent,
        refetchInterval: 30_000,
    });
}

export function useScoreDistribution(recognizer?: string) {
    return useQuery({
        queryKey: ["stats", "distribution", recognizer],
        queryFn: () => api.getDistribution(recognizer),
        refetchInterval: 60_000,
    });
}

export function useHighRiskEvents(limit = 50) {
    return useQuery({
        queryKey: ["stats", "high-risk", limit],
        queryFn: () => api.getHighRisk(limit),
        refetchInterval: 30_000,
    });
}

export function useEvents(params?: {
    recognizer?: string;
    flagged_only?: boolean;
    limit?: number;
    offset?: number;
}) {
    return useQuery({
        queryKey: ["stats", "events", params],
        queryFn: () => api.getEvents(params),
        refetchInterval: 15_000,
    });
}

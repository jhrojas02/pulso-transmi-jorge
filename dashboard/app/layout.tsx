import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pulso TransMi — Dashboard",
  description: "Observabilidad del pipeline: collector, champion, error por estación y leaderboard.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}

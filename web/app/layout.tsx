import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "StudioMyStock",
  description: "Turn raw car photos into studio-quality images",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}

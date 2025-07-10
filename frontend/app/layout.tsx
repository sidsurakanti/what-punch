import type { Metadata } from "next";
import "@/styles/globals.css";
import { sans, mono } from "@/styles/fonts";

export const metadata: Metadata = {
  title: "What punch?",
  description: "What punch?",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${sans.variable} ${mono.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}

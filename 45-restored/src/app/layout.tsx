export const metadata = {
  title: 'LyDian Trader API',
  description: 'AI-Powered Trading API Backend',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

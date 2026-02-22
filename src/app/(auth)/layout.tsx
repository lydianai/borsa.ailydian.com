// Auth layout - clean layout without sidebar for login/register
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'LyTrade Scanner',
  description: 'Kripto piyasalarında profesyonel sinyal analizi',
};

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Clean layout without sidebar, ErrorBoundary, etc.
  return <>{children}</>;
}

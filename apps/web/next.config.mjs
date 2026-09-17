/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  typedRoutes: true,
  async redirects() {
    return [{ source: '/market/crypto/coins', destination: '/market/crypto/coins/ZCAT', permanent: false }];
  }
};

export default nextConfig;

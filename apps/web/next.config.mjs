/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  typedRoutes: true,
  async redirects() {
    return [
      { source: '/market/crypto/people', destination: '/market/crypto?tab=people', permanent: false },
      { source: '/market/crypto/coins', destination: '/market/crypto?tab=timeline', permanent: false },
      { source: '/market/crypto/coins/:coin', destination: '/market/crypto?coin=:coin&tab=timeline', permanent: false },
      { source: '/market/crypto/accounts/:id', destination: '/market/crypto?account=:id', permanent: false },
      { source: '/market/crypto/data', destination: '/market/crypto?tab=data', permanent: false },
    ];
  }
};

export default nextConfig;

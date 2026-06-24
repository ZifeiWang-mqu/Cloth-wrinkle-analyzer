/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Allow <img> from the backend's /uploads static mount during dev.
  images: { unoptimized: true },
  // ESLint deps aren't bundled in this MVP; don't fail builds on lint.
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;

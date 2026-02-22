import { NextRequest, NextResponse } from 'next/server';

// IP Geolocation - Defensive security: Track login attempts
export async function GET(request: NextRequest) {
  try {
    // Get client IP from headers
    const forwarded = request.headers.get('x-forwarded-for');
    const realIp = request.headers.get('x-real-ip');
    const ip = forwarded?.split(',')[0] || realIp || '127.0.0.1';

    // Get user agent for device info
    const userAgent = request.headers.get('user-agent') || '';

    // Parse device info from user agent
    const deviceInfo = parseUserAgent(userAgent);

    // Use ipapi.co for geolocation (free, no API key required)
    let geoData;
    try {
      const geoResponse = await fetch(`https://ipapi.co/${ip}/json/`, {
        headers: {
          'User-Agent': 'LyTrade-Scanner/1.0'
        }
      });

      if (geoResponse.ok) {
        geoData = await geoResponse.json();
      } else {
        // Fallback data for development/localhost
        geoData = {
          ip: ip,
          city: 'Unknown',
          region: 'Unknown',
          country_name: 'Unknown',
          latitude: 41.0082,
          longitude: 28.9784,
          org: 'Unknown ISP',
          timezone: 'UTC'
        };
      }
    } catch (error) {
      // Fallback for localhost or API issues
      geoData = {
        ip: ip,
        city: 'Istanbul',
        region: 'Istanbul',
        country_name: 'Turkey',
        latitude: 41.0082,
        longitude: 28.9784,
        org: 'Development Environment',
        timezone: 'Europe/Istanbul'
      };
    }

    return NextResponse.json({
      success: true,
      data: {
        ip: ip,
        location: {
          city: geoData.city,
          region: geoData.region,
          country: geoData.country_name,
          latitude: geoData.latitude,
          longitude: geoData.longitude,
          timezone: geoData.timezone
        },
        device: {
          type: deviceInfo.type,
          os: deviceInfo.os,
          browser: deviceInfo.browser,
          brand: deviceInfo.brand
        },
        isp: geoData.org || 'Unknown ISP',
        timestamp: new Date().toISOString()
      }
    });

  } catch (error: any) {
    console.error('Geolocation error:', error);

    return NextResponse.json({
      success: false,
      error: 'Geolocation service unavailable',
      data: null
    }, {
      status: 500
    });
  }
}

// Parse user agent for device info
function parseUserAgent(userAgent: string) {
  const ua = userAgent.toLowerCase();

  // Device type
  let type = 'Desktop';
  if (ua.includes('mobile') || ua.includes('android') || ua.includes('iphone')) {
    type = 'Mobile';
  } else if (ua.includes('tablet') || ua.includes('ipad')) {
    type = 'Tablet';
  }

  // Operating System
  let os = 'Unknown';
  if (ua.includes('windows')) os = 'Windows';
  else if (ua.includes('mac os')) os = 'macOS';
  else if (ua.includes('linux')) os = 'Linux';
  else if (ua.includes('android')) os = 'Android';
  else if (ua.includes('ios') || ua.includes('iphone') || ua.includes('ipad')) os = 'iOS';

  // Browser
  let browser = 'Unknown';
  if (ua.includes('firefox')) browser = 'Firefox';
  else if (ua.includes('chrome')) browser = 'Chrome';
  else if (ua.includes('safari')) browser = 'Safari';
  else if (ua.includes('edge')) browser = 'Edge';
  else if (ua.includes('opera')) browser = 'Opera';

  // Brand (for mobile devices)
  let brand = 'Unknown';
  if (ua.includes('samsung')) brand = 'Samsung';
  else if (ua.includes('huawei')) brand = 'Huawei';
  else if (ua.includes('xiaomi')) brand = 'Xiaomi';
  else if (ua.includes('iphone') || ua.includes('ipad')) brand = 'Apple';
  else if (ua.includes('pixel')) brand = 'Google';
  else if (ua.includes('oneplus')) brand = 'OnePlus';

  return { type, os, browser, brand };
}

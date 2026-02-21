import { NextRequest, NextResponse } from "next/server";
const ENABLED=(process.env.NEXT_PUBLIC_PERSONAL_AUTH_ENABLED??"0")==="1";
const TOKEN=process.env.PERSONAL_AUTH_TOKEN??"";
export function personalAuthMiddleware(req:NextRequest){
  if(!ENABLED) return NextResponse.next();
  const url=new URL(req.url);
  const cand=req.headers.get("x-personal-auth")||url.searchParams.get("token")||"";
  if(TOKEN && cand===TOKEN) return NextResponse.next();
  return new NextResponse("Unauthorized",{status:401});
}

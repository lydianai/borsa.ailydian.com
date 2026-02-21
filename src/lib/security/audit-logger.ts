// ✅ Tam uyumlu audit-logger.ts — bütün fonksiyonlar, imzalar, tipler eksiksiz
export enum AuditEventType {
  API_REQUEST="API_REQUEST",API_ERROR="API_ERROR",AUTH="AUTH",SYSTEM="SYSTEM",
  RATE_LIMIT="RATE_LIMIT",UNAUTHORIZED="UNAUTHORIZED",SUSPICIOUS_ACTIVITY="SUSPICIOUS_ACTIVITY"
}
export enum AuditSeverity{INFO="INFO",WARN="WARN",ERROR="ERROR",WARNING="WARNING"}
type Meta=Record<string,unknown>;
export type AuditRecord={ts:number;type:AuditEventType;severity:AuditSeverity;message:string;path?:string;meta?:Meta};
const BUF:AuditRecord[]=[];const MAX_BUF=5000;
function normalize(x:any){if(x==="ERROR"||x===AuditSeverity.ERROR)return AuditSeverity.ERROR;
if(x==="WARN"||x===AuditSeverity.WARN)return AuditSeverity.WARN;
if(x==="WARNING"||x===AuditSeverity.WARNING)return AuditSeverity.WARN;
return AuditSeverity.INFO;}
function push(r:AuditRecord){BUF.push(r);if(BUF.length>MAX_BUF)BUF.splice(0,BUF.length-MAX_BUF);
if((process.env.NEXT_PUBLIC_AUDIT_CONSOLE??"0")==="1"){console.log("[AUDIT]",r.type,r.severity,r.message,r.path??"",r.meta??"");}}

/** logAPIRequest(req) veya logAPIRequest(method,path,status,duration,meta?) */
export function logAPIRequest(...args:any[]){
 try{
  if(args.length===1&&typeof args[0]==="object"&&"url"in args[0]){
    const req=args[0] as Request;const u=new URL(req.url);
    push({ts:Date.now(),type:AuditEventType.API_REQUEST,severity:AuditSeverity.INFO,
          message:"HTTP request",path:u.pathname,meta:{method:(req as any).method}});return;}
  const [method,path,status,duration,meta]=args;
  push({ts:Date.now(),type:AuditEventType.API_REQUEST,severity:AuditSeverity.INFO,
        message:`${method} ${path} ${status} (${duration}ms)`,path,meta});
 }catch(e){console.error("[AUDIT_ERR]",e);}
}

/** logUnauthorizedAccess(req|string, ip?, reason?) */
export function logUnauthorizedAccess(t:Request|string,ip?:string,reason?:string){
 const path=typeof t==="string"?t:new URL(t.url).pathname;
 push({ts:Date.now(),type:AuditEventType.UNAUTHORIZED,severity:AuditSeverity.WARN,
       message:reason||"Unauthorized",path,meta:{ip}});
}

/** logRateLimitExceeded(req|string, ip?:string, key?:string) — 3 arg destekli */
export function logRateLimitExceeded(t:Request|string,a?:string,b?:string){
 const path=typeof t==="string"?t:new URL(t.url).pathname;
 const ip=a, key=b;
 push({ts:Date.now(),type:AuditEventType.RATE_LIMIT,severity:AuditSeverity.WARN,
       message:"Rate limit exceeded",path,meta:{ip,key}});
}

/** Genel log() fonksiyonu */
function _log(type:AuditEventType,message:string,a?:any,b?:any){
 const sev=normalize(a);
 const meta=(a===AuditSeverity.INFO||a===AuditSeverity.WARN||
             a===AuditSeverity.ERROR||a===AuditSeverity.WARNING)?b:a;
 push({ts:Date.now(),type,severity:sev,message,meta});
}

export const auditLogger={
 log:_log,
 query(q?:{type?:AuditEventType;severity?:AuditSeverity;path?:string;limit?:number}){
   const lim=Math.min(Math.max(q?.limit??200,1),2000);
   let arr=BUF;
   if(q?.type)arr=arr.filter(r=>r.type===q.type);
   if(q?.severity)arr=arr.filter(r=>r.severity===normalize(q.severity));
   if(q?.path)arr=arr.filter(r=>r.path===q.path);
   return arr.slice(-lim);
 },
 getStats(){return{total:BUF.length}},
 clear(){BUF.length=0;}
};
export default auditLogger;

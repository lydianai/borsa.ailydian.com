from ailydian_client import AILYDIANBackend

backend = AILYDIANBackend(project_id="borsa.ailydian.com")
print("Backend connected:", backend.project_id)

# PyInstaller hook for fastapi
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all fastapi modules
datas, binaries, hiddenimports = collect_all('fastapi')

# Also collect uvicorn and pydantic
uvicorn_datas, uvicorn_binaries, uvicorn_imports = collect_all('uvicorn')
pydantic_datas, pydantic_binaries, pydantic_imports = collect_all('pydantic')
starlette_datas, starlette_binaries, starlette_imports = collect_all('starlette')

# Combine all
datas += uvicorn_datas + pydantic_datas + starlette_datas
binaries += uvicorn_binaries + pydantic_binaries + starlette_binaries
hiddenimports += uvicorn_imports + pydantic_imports + starlette_imports


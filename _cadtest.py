"""CAD runner 离线自测（需先 `pip install build123d`）。

用法：
    DATABASE_URL=... S3_ENDPOINT=... S3_ACCESS_KEY=x S3_SECRET_KEY=y ADMIN_PASSWORD=z \
        python _cadtest.py

验证：固定的 build123d 代码能被子进程执行并导出 STEP/STL/GLB，
且 GLB 以 glTF 二进制魔数 'glTF' 开头。同时验证危险代码被静态拦截。
"""
import asyncio

from app import cad_runner

GOOD = "result = Box(20, 20, 20)"
BAD = "import os\nos.system('echo hi')\nresult = 1"


async def main() -> None:
    # 1) 危险代码必须被拦截
    try:
        await cad_runner.run_build123d_script(BAD)
        print("FAIL: 危险代码未被拦截")
        return
    except cad_runner.CadRunError as e:
        print("[ok] 危险代码被拦截:", e)

    # 2) 正常代码导出三种格式
    out = await cad_runner.run_build123d_script(GOOD)
    for fmt in ("step", "stl", "glb"):
        assert fmt in out and out[fmt], f"缺少 {fmt}"
        print(f"[ok] {fmt}: {len(out[fmt])} bytes")
    assert out["glb"][:4] == b"glTF", "GLB 魔数不正确"
    print("[ok] GLB 魔数正确")
    print("ALL PASS")


if __name__ == "__main__":
    asyncio.run(main())

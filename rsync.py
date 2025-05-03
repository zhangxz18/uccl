from shared import *


async def rsync(nodes):
    folders = ["/opt/uccl_rdma/", "/opt/uccl_rdma_a2a/", "/opt/uccl_rdma_ar/"]
    tasks = []
    for folder in folders:
        tasks.extend(
            [
                run_command(
                    f"rsync -auv -e 'ssh -o StrictHostKeyChecking=no' {folder} {node}:{folder}"
                )
                for node in nodes
            ]
        )

    # Run all tasks in parallel
    await asyncio.gather(*tasks)


nodes = get_nodes()
print(f"Nodes: {nodes}")

asyncio.run(rsync(nodes))

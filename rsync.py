from shared import *


async def rsync(nodes):
    tasks = [
        run_command(
            f"rsync -auv -e 'ssh -o StrictHostKeyChecking=no' /opt/uccl_rdma/ {node}:/opt/uccl_rdma/"
        )
        for node in nodes
    ]

    # Run all tasks in parallel
    await asyncio.gather(*tasks)


nodes = get_nodes()
print(f"Nodes: {nodes}")

asyncio.run(rsync(nodes))

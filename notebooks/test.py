import brayns

CA = '/gpfs/bbp.cscs.ch/project/proj112/home/kurban/topology_paper/toolbox/graph_analysis/notebooks/CA.crt'
URI = 'r1i4n13.bbp.epfl.ch:5000'
SSL = brayns.SslClientContext(CA)

connector = brayns.Connector(URI, SSL)

with connector.connect() as instance:
    
    print(brayns.get_models(instance))
    print(brayns.remove_models(instance, [2]))
    
    capsules = [
        (
            brayns.Capsule(
                brayns.Vector3(1, 1, 1),
                1,
                brayns.Vector3(2, 2, 2),
                0
            ),
            brayns.Color4(1, 1, 1)
        ),
    ]
    brayns.add_geometries(instance, capsules)
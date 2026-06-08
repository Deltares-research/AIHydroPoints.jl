const R=6378137 # Earth semi-major axis in meter

function distance(point1, point2)
    lat1, lon1 = point1
    lat2, lon2 = point2
    delta_lon = abs(lon1-lon2)
    delta_sig = acos(clamp(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(delta_lon), -1.0, 1.0))
    return R*delta_sig
end

# function get_distances(in_points, out_points)
#     distances = distance.(in_points, reshape(out_points, (1,:)))
#     return distances
# end

function get_adjacency(in_points, out_points; max_distance=1e5)
    distances = distance.(in_points, reshape(out_points, (1,:)))
    return distances .<= max_distance
end

mutable struct GraphNetwork
    in_points
    out_points
    adjacency 
end

function GraphNetwork(in_points, out_points; max_distance=1e5, format="deg")
    inputs = in_points
    outputs = out_points
    if format == "deg"
        inputs = [deg2rad.(point) for point in inputs]
        outputs = [deg2rad.(point) for point in outputs]
    end
    adjacency = get_adjacency(inputs, outputs; max_distance=max_distance)

    isolated = sum(adjacency, dims=1) .== 0

    if any(isolated)
        error("Output points $(findall(isolated[:])) are not connected to any input point")
    end

    return GraphNetwork(in_points, out_points, adjacency)
end


function plot_graph(gn::GraphNetwork)

    function draw_edge(pl, point1, point2)
        plot!(pl, [point1[1], point2[1]], [point1[2], point2[2]], permute=(:x,:y),
            arrow=:arrow, label=false, color=:black, lw=2)
    end

    pl = scatter(gn.in_points, permute=(:x,:y), label="Input Points",
        markersize=7, markercolor=:red)
    scatter!(pl, gn.out_points, permute=(:x,:y), label="Output Points",
        markersize=7, markercolor=:blue)

    for idx in CartesianIndices(gn.adjacency)
        if gn.adjacency[idx]
            draw_edge(pl, gn.in_points[idx[1]], gn.out_points[idx[2]])
        end
    end

    return pl

end
module EscherCoreGeneratorTrippleJulia

function compareTwoCoreElements(a, b)
    if a < b
        return LESS
    end
    return GREATER
end

function getCoreComparisions(partition)
    return Dict(
        "0" => ["subescher uv start", "subescher uw start", "subescher vw start", "subescher uv_w start", "subescher uw_v start", "subescher vw_u start"],
        "subescher uv end" => ["uv 1. insert", "uw 1. insert", "vw 1. insert"],
        "subescher uw end" => ["uv 1. insert", "uw 1. insert", "vw 1. insert"],
        "subescher vw end" => ["uv 1. insert", "uw 1. insert", "vw 1. insert"]
    )
end

function getCoreLabels(partition)
    return ["0", "subescher uv start", "subescher uv end", "uv 1. insert", "subescher vw start", "subescher vw end", "vw 1. insert", 
            "subescher uw start", "subescher uw end", "uw 1. insert", "subescher uv_w start", 
            "subescher uv_w end", "uv_w 1. insert", "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert",
            "subescher vw_u start", "subescher vw_u end", "vw_u 1. insert"]
end

function getCoreLabelGroups(partition)
    return Dict(
        "0" => Dict("0" => "0"),
        "subescher start" => Dict(
            "subescher uv start" => "uv",
            "subescher vw start" => "vw",
            "subescher uw start" => "uw",
            "subescher uv_w start" => "uv_w",
            "subescher uw_v start" => "uw_v",
            "subescher vw_u start" => "vw_u"
        ),
        "subescher end" => Dict(
            "subescher uv end" => "uv",
            "subescher vw end" => "vw",
            "subescher uw end" => "uw",
            "subescher uv_w end" => "uv_w",
            "subescher uw_v end" => "uw_v",
            "subescher vw_u end" => "vw_u"
        ),
        "1. insertion" => Dict(
            "uv 1. insert" => "uv",
            "vw 1. insert" => "vw",
            "uw 1. insert" => "uw",
            "uv_w 1. insert" => "uv_w",
            "uw_v 1. insert" => "uw_v",
            "vw_u 1. insert" => "vw_u"
        )
    )
end

function getCoreLabelGroupColors(partition)
    return Dict(
        "0" => "skyblue",
        "subescher start" => "hotpink",
        "subescher end" => "silver",
        "1. insertion" => "limegreen"
    )
end

function generateCore(self, eschertripple)
    function add_half(L)
        return vcat(L[1:end-1], L[end] + 0.5)
    end
    u, v, w = eschertripple

    core_u_v = self.get_shortb_insertion_and_subescher_of_2_eschers(u, v)
    core_v_w = self.get_shortb_insertion_and_subescher_of_2_eschers(v, w)
    core_u_w = self.get_shortb_insertion_and_subescher_of_2_eschers(u, w)

    if core_u_v[end] == -1
        core_uv_w = [-1, -1, -1]
    else
        uv = self.concat(u, v, insertionpoint=core_u_v[end])
        core_uv_w = self.get_shortb_insertion_and_subescher_of_2_eschers(uv, w)
    end

    if core_u_w[end] == -1
        core_uw_v = [-1, -1, -1]
    else
        uw = self.concat(u, w, insertionpoint=core_u_w[end])
        core_uw_v = self.get_shortb_insertion_and_subescher_of_2_eschers(uw, v)
    end

    if core_v_w[end] == -1
        core_vw_u = [-1, -1, -1]
    else
        vw = self.concat(v, w, insertionpoint=core_v_w[end])
        core_vw_u = self.get_shortb_insertion_and_subescher_of_2_eschers(vw, u)
    end

    return vcat([0], add_half(core_u_v), add_half(core_v_w), add_half(core_u_w), add_half(core_uv_w), add_half(core_uw_v), add_half(core_vw_u))
end

function get_insertion_and_subescher_of_2_eschers(self, u, v)
    n, k = length(u), length(v)
    core = self.getInsertionsSubeshers(u, v)
    insertions, escherstartpoints = core

    if length(insertions) == 0
        return [n-1, -1, -1, -1]
    else
        points = [n-1]
        has_bigger_than_0_startpoint = false
        for escherstartpoint in escherstartpoints
            if escherstartpoint > 0
                push!(points, escherstartpoint)
                push!(points, escherstartpoint + k - 1)
                has_bigger_than_0_startpoint = true
                break
            end
        end
        if !has_bigger_than_0_startpoint
            push!(points, -1)
            push!(points, -2 + k)
        end
        push!(points, insertions[1])
    end
    return points
end

function get_short_insertion_and_subescher_of_2_eschers(self, u, v)
    n, k = length(u), length(v)
    core = self.getInsertionsSubeshers(u, v)
    insertions, escherstartpoints = core

    if length(insertions) == 0
        return [-1, -1, -1]
    else
        points = []
        has_bigger_than_0_startpoint = false
        for escherstartpoint in escherstartpoints
            if escherstartpoint > 0
                push!(points, escherstartpoint)
                push!(points, escherstartpoint + k - 1)
                has_bigger_than_0_startpoint = true
                break
            end
        end
        if !has_bigger_than_0_startpoint
            push!(points, -1)
            push!(points, k - 2)
        end
        push!(points, insertions[1])
    end
    return points
end

function get_shortb_insertion_and_subescher_of_2_eschers(self, u, v)
    n, k = length(u), length(v)
    core = self.getInsertionsSubeshers(u, v)
    insertions, escherstartpoints = core

    if length(insertions) == 0
        return [-1, -1, -1]
    else
        points = []
        has_bigger_than_0_startpoint = false
        for escherstartpoint in escherstartpoints
            if escherstartpoint > 0
                push!(points, escherstartpoint)
                push!(points, escherstartpoint + k - 1)
                has_bigger_than_0_startpoint = true
                break
            end
        end
        if !has_bigger_than_0_startpoint
            push!(points, -1)
            push!(points, insertions[1] + 1)
        end
        push!(points, insertions[1])
    end
    return points
end

function generateCore_2partition(self, escherpair)
    n, k = self.partition
    u, v = escherpair
    core = self.getInsertionsSubeshers(u, v)
    insertions, escherstartpoints = core

    if length(insertions) == 0
        return []
    else
        points = [0]
        has_bigger_than_0_startpoint = false
        for escherstartpoint in escherstartpoints
            if escherstartpoint > 0
                push!(points, escherstartpoint)
                push!(points, escherstartpoint + k - 1)
                has_bigger_than_0_startpoint = true
                break
            end
        end
        if !has_bigger_than_0_startpoint
            push!(points, -1)
            push!(points, -2 + k)
        end
        push!(points, insertions[1])
        push!(points, n - 1)
    end
    return points
end

function concat(self, first, second, insertionpoint)
    f = length(first)
    k = length(second)
    return vcat(first[1:insertionpoint % f + 1], self.cyclicslice(second, insertionpoint + 1, insertionpoint + k + 1), first[insertionpoint % f + 2:end])
end

function getInsertionsSubeshers(self, u, v)
    n = length(u)
    k = length(v)
    lcm = lcm(n, k)
    uu = repeat(u, lcm ÷ n)
    insertions = self.getInsertionPoints(u, v, lcm)
    subeschers = []
    if length(insertions) > 0
        insertion = insertions[1]
        extrav = self.cyclicslice(v, insertion + 1, insertion + 2)
        groundtrooper = self.getpseudosubescherstartingpoints(u, k)
        snakefan = self.getpseudosubescherstartingpoints(vcat(uu[1:insertion + 1], extrav), k)
        subeschers = snakefan
    end
    return (insertions, subeschers)
end

function getInsertionPoints(self, u, v, lcm=0)
    n = length(u)
    k = length(v)
    if lcm == 0
        lcm = lcm(n, k)
    end
    points = []
    for i in 0:lcm-1
        if self.uio.comparison_matrix[u[i % n + 1], v[(i + 1) % k + 1]] != UIO.GREATER && self.uio.comparison_matrix[v[i % k + 1], u[(i + 1) % n + 1]] != UIO.GREATER
            push!(points, i)
        end
    end
    return points
end

function cyclicslice(tuple, start, stop)
    n = length(tuple)
    start = start % n
    stop = stop % n
    if start < stop
        return tuple[start+1:stop]
    elseif start == stop
        return tuple
    end
    return vcat(tuple[start+1:end], tuple[1:stop])
end

function getpseudosubescherstartingpoints(self, escher, k)
    # k > 0, pretends the input is an escher and finds valid k-subescher
    subeschersstartingpoint = Int[]  # start of the box
    h = length(escher)
    # println("escher length:", h)
    for m in -1:(h-2)  # m can be 0, m is before the start of the box mBBBB#
        cond1 = self.uio.isarrow(escher, (m+k) % h, (m+1) % h)
        cond2 = self.uio.isarrow(escher, m % h, (m+k+1) % h)
        # if m == 1
        #     println((m+k) % h, escher[(m+k) % h])
        #     println((m+1) % h, escher[(m+1) % h])
        #     println(m, escher[m])
        #     println((m+k+1) % h, escher[(m+k+1) % h])
        # println("m:", m)
        # println("cond1:", cond1)
        # println("cond2:", cond2)
        if cond1 && cond2
            push!(subeschersstartingpoint, m+1)
        end
    end
    return subeschersstartingpoint
end

end  # module EscherCoreGeneratorTrippleSymmetricNoEqual
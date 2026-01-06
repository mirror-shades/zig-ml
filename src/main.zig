const std = @import("std");

const MAX_MODEL_VAR_INPUTS = 2;

const MV = struct {
    pub const none: u32 = 0;
    pub const requires_grad: u32 = 1 << 0;
    pub const parameter: u32 = 1 << 1;
    pub const input: u32 = 1 << 2;
    pub const output: u32 = 1 << 3;
    pub const desired_output: u32 = 1 << 4;
    pub const cost: u32 = 1 << 5;
};

const ModelVarOp = enum(u8) {
    null,
    create,

    unary_start,
    relu,
    softmax,

    binary_start,
    add,
    sub,
    matmul,
    cross_entropy,
};

fn mvNumInputs(op: ModelVarOp) u32 {
    return if (@intFromEnum(op) < @intFromEnum(ModelVarOp.unary_start))
        0
    else if (@intFromEnum(op) < @intFromEnum(ModelVarOp.binary_start))
        1
    else
        2;
}

const Matrix = struct {
    rows: u32,
    cols: u32,
    data: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, rows: u32, cols: u32) !Matrix {
        const len: usize = @intCast(@as(u64, rows) * @as(u64, cols));
        const data = try allocator.alloc(f32, len);
        @memset(data, 0.0);
        return .{ .rows = rows, .cols = cols, .data = data, .allocator = allocator };
    }

    pub fn deinit(self: *Matrix) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }

    pub fn clear(self: *Matrix) void {
        @memset(self.data, 0.0);
    }

    pub fn fill(self: *Matrix, x: f32) void {
        @memset(self.data, x);
    }

    pub fn scale(self: *Matrix, s: f32) void {
        for (self.data) |*v| v.* *= s;
    }

    pub fn sum(self: *const Matrix) f32 {
        var s: f32 = 0.0;
        for (self.data) |v| s += v;
        return s;
    }

    pub fn argmax(self: *const Matrix) u32 {
        var max_i: usize = 0;
        for (self.data, 0..) |v, i| {
            if (v > self.data[max_i]) max_i = i;
        }
        return @intCast(max_i);
    }
};

fn matAdd(out: *Matrix, a: *const Matrix, b: *const Matrix) !void {
    if (a.rows != b.rows or a.cols != b.cols) return error.InvalidDimensions;
    if (out.rows != a.rows or out.cols != a.cols) return error.InvalidDimensions;
    for (out.data, 0..) |*v, i| v.* = a.data[i] + b.data[i];
}

fn matSub(out: *Matrix, a: *const Matrix, b: *const Matrix) !void {
    if (a.rows != b.rows or a.cols != b.cols) return error.InvalidDimensions;
    if (out.rows != a.rows or out.cols != a.cols) return error.InvalidDimensions;
    for (out.data, 0..) |*v, i| v.* = a.data[i] - b.data[i];
}

fn matRelu(out: *Matrix, in: *const Matrix) !void {
    if (out.rows != in.rows or out.cols != in.cols) return error.InvalidDimensions;
    for (out.data, 0..) |*v, i| v.* = @max(0.0, in.data[i]);
}

fn matSoftmax(out: *Matrix, in: *const Matrix) !void {
    if (out.rows != in.rows or out.cols != in.cols) return error.InvalidDimensions;
    var sum: f32 = 0.0;
    for (out.data, 0..) |*v, i| {
        const e = @exp(in.data[i]);
        v.* = e;
        sum += e;
    }
    if (sum != 0.0) out.scale(1.0 / sum);
}

fn matCrossEntropy(out: *Matrix, p: *const Matrix, q: *const Matrix) !void {
    if (p.rows != q.rows or p.cols != q.cols) return error.InvalidDimensions;
    if (out.rows != p.rows or out.cols != p.cols) return error.InvalidDimensions;
    for (out.data, 0..) |*v, i| {
        const pi = p.data[i];
        v.* = if (pi == 0.0) 0.0 else pi * -@log(q.data[i]);
    }
}

fn matMul(out: *Matrix, a: *const Matrix, b: *const Matrix, zero_out: bool, transpose_a: bool, transpose_b: bool) !void {
    const a_rows: u32 = if (transpose_a) a.cols else a.rows;
    const a_cols: u32 = if (transpose_a) a.rows else a.cols;
    const b_rows: u32 = if (transpose_b) b.cols else b.rows;
    const b_cols: u32 = if (transpose_b) b.rows else b.cols;

    if (a_cols != b_rows) return error.InvalidDimensions;
    if (out.rows != a_rows or out.cols != b_cols) return error.InvalidDimensions;

    if (zero_out) out.clear();

    const transpose_code: u2 = (@as(u2, @intFromBool(transpose_a)) << 1) | @as(u2, @intFromBool(transpose_b));
    switch (transpose_code) {
        0b00 => { // nn
            for (0..out.rows) |i| {
                for (0..a.cols) |k| {
                    const a_ik = a.data[k + i * a.cols];
                    for (0..out.cols) |j| {
                        out.data[j + i * out.cols] += a_ik * b.data[j + k * b.cols];
                    }
                }
            }
        },
        0b01 => { // nt
            for (0..out.rows) |i| {
                for (0..out.cols) |j| {
                    var sum: f32 = 0.0;
                    for (0..a.cols) |k| {
                        sum += a.data[k + i * a.cols] * b.data[k + j * b.cols];
                    }
                    out.data[j + i * out.cols] += sum;
                }
            }
        },
        0b10 => { // tn
            for (0..a.rows) |k| {
                for (0..out.rows) |i| {
                    const a_ik = a.data[i + k * a.cols];
                    for (0..out.cols) |j| {
                        out.data[j + i * out.cols] += a_ik * b.data[j + k * b.cols];
                    }
                }
            }
        },
        0b11 => { // tt
            for (0..out.rows) |i| {
                for (0..out.cols) |j| {
                    var sum: f32 = 0.0;
                    for (0..a.rows) |k| {
                        sum += a.data[i + k * a.cols] * b.data[k + j * b.cols];
                    }
                    out.data[j + i * out.cols] += sum;
                }
            }
        },
    }
}

fn matReluAddGrad(out: *Matrix, in: *const Matrix, grad: *const Matrix) !void {
    if (out.rows != in.rows or out.cols != in.cols) return error.InvalidDimensions;
    if (out.rows != grad.rows or out.cols != grad.cols) return error.InvalidDimensions;
    for (out.data, 0..) |*v, i| v.* += if (in.data[i] > 0.0) grad.data[i] else 0.0;
}

fn matSoftmaxAddGrad(out: *Matrix, softmax_out: *const Matrix, grad: *const Matrix, allocator: std.mem.Allocator) !void {
    if (!((softmax_out.rows == 1) or (softmax_out.cols == 1))) return error.InvalidDimensions;
    const size: u32 = @max(softmax_out.rows, softmax_out.cols);

    var jacobian = try Matrix.init(allocator, size, size);
    defer jacobian.deinit();

    for (0..size) |i| {
        for (0..size) |j| {
            jacobian.data[j + i * size] = softmax_out.data[i] * (@as(f32, @floatFromInt(@intFromBool(i == j))) - softmax_out.data[j]);
        }
    }

    try matMul(out, &jacobian, grad, true, false, false);
}

fn matCrossEntropyAddGrad(p_grad: ?*Matrix, q_grad: ?*Matrix, p: *const Matrix, q: *const Matrix, grad: *const Matrix) !void {
    if (p.rows != q.rows or p.cols != q.cols) return error.InvalidDimensions;
    const size: usize = p.data.len;

    if (p_grad) |pg| {
        if (pg.rows != p.rows or pg.cols != p.cols) return error.InvalidDimensions;
        for (0..size) |i| pg.data[i] += -@log(q.data[i]) * grad.data[i];
    }

    if (q_grad) |qg| {
        if (qg.rows != q.rows or qg.cols != q.cols) return error.InvalidDimensions;
        for (0..size) |i| qg.data[i] += -p.data[i] / q.data[i] * grad.data[i];
    }
}

const ModelVar = struct {
    index: u32,
    flags: u32,

    val: *Matrix,
    grad: ?*Matrix,

    op: ModelVarOp,
    inputs: [MAX_MODEL_VAR_INPUTS]?*ModelVar,
};

const ModelProgram = struct {
    vars: []*ModelVar,
};

const ModelContext = struct {
    num_vars: u32 = 0,

    input: ?*ModelVar = null,
    output: ?*ModelVar = null,
    desired_output: ?*ModelVar = null,
    cost: ?*ModelVar = null,

    forward_prog: ?ModelProgram = null,
    cost_prog: ?ModelProgram = null,
};

fn mvCreate(allocator: std.mem.Allocator, model: *ModelContext, rows: u32, cols: u32, flags: u32) !*ModelVar {
    const node = try allocator.create(ModelVar);
    errdefer allocator.destroy(node);

    const val = try allocator.create(Matrix);
    errdefer allocator.destroy(val);
    val.* = try Matrix.init(allocator, rows, cols);

    var grad: ?*Matrix = null;
    if ((flags & MV.requires_grad) != 0) {
        const g = try allocator.create(Matrix);
        errdefer allocator.destroy(g);
        g.* = try Matrix.init(allocator, rows, cols);
        grad = g;
    }

    node.* = .{
        .index = model.num_vars,
        .flags = flags,
        .val = val,
        .grad = grad,
        .op = .create,
        .inputs = .{ null, null },
    };
    model.num_vars += 1;

    if ((flags & MV.input) != 0) model.input = node;
    if ((flags & MV.output) != 0) model.output = node;
    if ((flags & MV.desired_output) != 0) model.desired_output = node;
    if ((flags & MV.cost) != 0) model.cost = node;

    return node;
}

fn mvUnary(allocator: std.mem.Allocator, model: *ModelContext, input: *ModelVar, flags: u32, op: ModelVarOp) !*ModelVar {
    var out_flags = flags;
    if ((input.flags & MV.requires_grad) != 0) out_flags |= MV.requires_grad;
    const out = try mvCreate(allocator, model, input.val.rows, input.val.cols, out_flags);
    out.op = op;
    out.inputs[0] = input;
    return out;
}

fn mvBinary(allocator: std.mem.Allocator, model: *ModelContext, a: *ModelVar, b: *ModelVar, rows: u32, cols: u32, flags: u32, op: ModelVarOp) !*ModelVar {
    var out_flags = flags;
    if (((a.flags | b.flags) & MV.requires_grad) != 0) out_flags |= MV.requires_grad;
    const out = try mvCreate(allocator, model, rows, cols, out_flags);
    out.op = op;
    out.inputs[0] = a;
    out.inputs[1] = b;
    return out;
}

fn mvRelu(allocator: std.mem.Allocator, model: *ModelContext, input: *ModelVar, flags: u32) !*ModelVar {
    return mvUnary(allocator, model, input, flags, .relu);
}

fn mvSoftmax(allocator: std.mem.Allocator, model: *ModelContext, input: *ModelVar, flags: u32) !*ModelVar {
    return mvUnary(allocator, model, input, flags, .softmax);
}

fn mvAdd(allocator: std.mem.Allocator, model: *ModelContext, a: *ModelVar, b: *ModelVar, flags: u32) !*ModelVar {
    if (a.val.rows != b.val.rows or a.val.cols != b.val.cols) return error.InvalidDimensions;
    return mvBinary(allocator, model, a, b, a.val.rows, a.val.cols, flags, .add);
}

fn mvSub(allocator: std.mem.Allocator, model: *ModelContext, a: *ModelVar, b: *ModelVar, flags: u32) !*ModelVar {
    if (a.val.rows != b.val.rows or a.val.cols != b.val.cols) return error.InvalidDimensions;
    return mvBinary(allocator, model, a, b, a.val.rows, a.val.cols, flags, .sub);
}

fn mvMatmul(allocator: std.mem.Allocator, model: *ModelContext, a: *ModelVar, b: *ModelVar, flags: u32) !*ModelVar {
    if (a.val.cols != b.val.rows) return error.InvalidDimensions;
    return mvBinary(allocator, model, a, b, a.val.rows, b.val.cols, flags, .matmul);
}

fn mvCrossEntropy(allocator: std.mem.Allocator, model: *ModelContext, p: *ModelVar, q: *ModelVar, flags: u32) !*ModelVar {
    if (p.val.rows != q.val.rows or p.val.cols != q.val.cols) return error.InvalidDimensions;
    return mvBinary(allocator, model, p, q, p.val.rows, p.val.cols, flags, .cross_entropy);
}

fn modelProgramCreate(allocator: std.mem.Allocator, model: *const ModelContext, out_var: *ModelVar) !ModelProgram {
    var scratch = std.heap.ArenaAllocator.init(allocator);
    defer scratch.deinit();
    const sa = scratch.allocator();

    var visited = try sa.alloc(bool, model.num_vars);
    @memset(visited, false);

    var stack = try sa.alloc(*ModelVar, model.num_vars);
    var out = try sa.alloc(*ModelVar, model.num_vars);

    var stack_size: usize = 0;
    var out_size: usize = 0;
    stack[stack_size] = out_var;
    stack_size += 1;

    while (stack_size > 0) {
        stack_size -= 1;
        const cur = stack[stack_size];

        if (cur.index >= model.num_vars) continue;

        if (visited[cur.index]) {
            out[out_size] = cur;
            out_size += 1;
            continue;
        }

        visited[cur.index] = true;

        stack[stack_size] = cur;
        stack_size += 1;

        const num_inputs = mvNumInputs(cur.op);
        var i: u32 = 0;
        while (i < num_inputs) : (i += 1) {
            const maybe_input = cur.inputs[i];
            if (maybe_input == null) continue;
            const input = maybe_input.?;
            if (input.index >= model.num_vars or visited[input.index]) continue;

            var j: usize = 0;
            while (j < stack_size) : (j += 1) {
                if (stack[j] == input) {
                    var k: usize = j;
                    while (k + 1 < stack_size) : (k += 1) stack[k] = stack[k + 1];
                    stack_size -= 1;
                    break;
                }
            }

            stack[stack_size] = input;
            stack_size += 1;
        }
    }

    const program_vars = try allocator.alloc(*ModelVar, out_size);
    @memcpy(program_vars, out[0..out_size]);
    return .{ .vars = program_vars };
}

fn modelProgramCompute(program: *const ModelProgram) !void {
    for (program.vars) |cur| {
        const a = cur.inputs[0];
        const b = cur.inputs[1];
        switch (cur.op) {
            .null, .create, .unary_start, .binary_start => {},
            .relu => try matRelu(cur.val, a.?.val),
            .softmax => try matSoftmax(cur.val, a.?.val),
            .add => try matAdd(cur.val, a.?.val, b.?.val),
            .sub => try matSub(cur.val, a.?.val, b.?.val),
            .matmul => try matMul(cur.val, a.?.val, b.?.val, true, false, false),
            .cross_entropy => try matCrossEntropy(cur.val, a.?.val, b.?.val),
        }
    }
}

fn modelProgramComputeGrads(program: *const ModelProgram, allocator: std.mem.Allocator) !void {
    for (program.vars) |cur| {
        if ((cur.flags & MV.requires_grad) == 0) continue;
        if ((cur.flags & MV.parameter) != 0) continue;
        if (cur.grad) |g| g.clear();
    }

    const last = program.vars[program.vars.len - 1];
    if (last.grad == null) return error.MissingGradient;
    last.grad.?.fill(1.0);

    var i: usize = program.vars.len;
    while (i > 0) {
        i -= 1;
        const cur = program.vars[i];
        if ((cur.flags & MV.requires_grad) == 0) continue;
        if (cur.grad == null) continue;

        const a = cur.inputs[0];
        const b = cur.inputs[1];
        const num_inputs = mvNumInputs(cur.op);

        if (num_inputs == 1 and (a.?.flags & MV.requires_grad) == 0) continue;
        if (num_inputs == 2 and (a.?.flags & MV.requires_grad) == 0 and (b.?.flags & MV.requires_grad) == 0) continue;

        switch (cur.op) {
            .null, .create, .unary_start, .binary_start => {},
            .relu => if ((a.?.flags & MV.requires_grad) != 0) try matReluAddGrad(a.?.grad.?, a.?.val, cur.grad.?),
            .softmax => if ((a.?.flags & MV.requires_grad) != 0) try matSoftmaxAddGrad(a.?.grad.?, cur.val, cur.grad.?, allocator),
            .add => {
                if ((a.?.flags & MV.requires_grad) != 0) try matAdd(a.?.grad.?, a.?.grad.?, cur.grad.?);
                if ((b.?.flags & MV.requires_grad) != 0) try matAdd(b.?.grad.?, b.?.grad.?, cur.grad.?);
            },
            .sub => {
                if ((a.?.flags & MV.requires_grad) != 0) try matAdd(a.?.grad.?, a.?.grad.?, cur.grad.?);
                if ((b.?.flags & MV.requires_grad) != 0) try matSub(b.?.grad.?, b.?.grad.?, cur.grad.?);
            },
            .matmul => {
                if ((a.?.flags & MV.requires_grad) != 0) try matMul(a.?.grad.?, cur.grad.?, b.?.val, false, false, true);
                if ((b.?.flags & MV.requires_grad) != 0) try matMul(b.?.grad.?, a.?.val, cur.grad.?, false, true, false);
            },
            .cross_entropy => try matCrossEntropyAddGrad(a.?.grad, b.?.grad, a.?.val, b.?.val, cur.grad.?),
        }
    }
}

fn modelCompile(allocator: std.mem.Allocator, model: *ModelContext) !void {
    if (model.output) |output| model.forward_prog = try modelProgramCreate(allocator, model, output);
    if (model.cost) |cost| model.cost_prog = try modelProgramCreate(allocator, model, cost);
}

fn modelFeedForward(model: *const ModelContext) !void {
    if (model.forward_prog) |prog| try modelProgramCompute(&prog);
}

const ModelTrainingDesc = struct {
    train_images: *Matrix,
    train_labels: *Matrix,
    test_images: *Matrix,
    test_labels: *Matrix,

    epochs: u32,
    batch_size: u32,
    learning_rate: f32,
};

fn modelTrain(model: *const ModelContext, training_desc: *const ModelTrainingDesc, allocator: std.mem.Allocator, rng: std.Random) !void {
    const train_images = training_desc.train_images;
    const train_labels = training_desc.train_labels;
    const test_images = training_desc.test_images;
    const test_labels = training_desc.test_labels;

    const input_node = model.input orelse return error.MissingInput;
    const output_node = model.output orelse return error.MissingOutput;
    const y_node = model.desired_output orelse return error.MissingDesiredOutput;
    const cost_node = model.cost orelse return error.MissingCost;
    const cost_prog = model.cost_prog orelse return error.MissingProgram;

    const num_examples = train_images.rows;
    const input_size = train_images.cols;
    const output_size = train_labels.cols;
    const num_tests = test_images.rows;
    const num_batches = num_examples / training_desc.batch_size;

    var scratch = std.heap.ArenaAllocator.init(allocator);
    defer scratch.deinit();
    const sa = scratch.allocator();

    var training_order = try sa.alloc(u32, num_examples);
    for (0..num_examples) |i| training_order[i] = @intCast(i);

    for (0..training_desc.epochs) |epoch| {
        for (0..num_examples) |_| {
            const a = rng.intRangeLessThan(u32, 0, num_examples);
            const b = rng.intRangeLessThan(u32, 0, num_examples);
            const tmp = training_order[b];
            training_order[b] = training_order[a];
            training_order[a] = tmp;
        }

        for (0..num_batches) |batch| {
            for (cost_prog.vars) |cur| {
                if ((cur.flags & MV.parameter) == 0) continue;
                if (cur.grad) |g| g.clear();
            }

            var avg_cost: f32 = 0.0;
            for (0..training_desc.batch_size) |i| {
                const order_index: u32 = @intCast(batch * training_desc.batch_size + i);
                const index = training_order[order_index];

                const in_off: usize = @intCast(@as(u64, index) * @as(u64, input_size));
                @memcpy(input_node.val.data[0..@intCast(input_size)], train_images.data[in_off .. in_off + @as(usize, input_size)]);

                const y_off: usize = @intCast(@as(u64, index) * @as(u64, output_size));
                @memcpy(y_node.val.data[0..@intCast(output_size)], train_labels.data[y_off .. y_off + @as(usize, output_size)]);

                try modelProgramCompute(&cost_prog);
                try modelProgramComputeGrads(&cost_prog, allocator);
                avg_cost += cost_node.val.sum();
            }
            avg_cost /= @floatFromInt(training_desc.batch_size);

            for (cost_prog.vars) |cur| {
                if ((cur.flags & MV.parameter) == 0) continue;
                const g = cur.grad orelse continue;
                g.scale(training_desc.learning_rate / @as(f32, @floatFromInt(training_desc.batch_size)));
                try matSub(cur.val, cur.val, g);
            }

            std.debug.print("Epoch {d:2}/{d:2} Batch {d:4}/{d:4} AvgCost {d:.4}\r", .{ epoch + 1, training_desc.epochs, batch + 1, num_batches, avg_cost });
        }
        std.debug.print("\n", .{});

        var num_correct: u32 = 0;
        var avg_cost: f32 = 0.0;
        for (0..num_tests) |i| {
            const idx: u32 = @intCast(i);
            const in_off: usize = @intCast(@as(u64, idx) * @as(u64, input_size));
            @memcpy(input_node.val.data[0..@intCast(input_size)], test_images.data[in_off .. in_off + @as(usize, input_size)]);

            const y_off: usize = @intCast(@as(u64, idx) * @as(u64, output_size));
            @memcpy(y_node.val.data[0..@intCast(output_size)], test_labels.data[y_off .. y_off + @as(usize, output_size)]);

            try modelProgramCompute(&cost_prog);
            avg_cost += cost_node.val.sum();

            num_correct += @intFromBool(output_node.val.argmax() == y_node.val.argmax());
        }

        avg_cost /= @floatFromInt(num_tests);
        const acc_pct = @as(f32, @floatFromInt(num_correct)) / @as(f32, @floatFromInt(num_tests)) * 100.0;
        std.debug.print("Test accuracy {d}/{d} ({d:.1}%) avgCost {d:.4}\n", .{ num_correct, num_tests, acc_pct, avg_cost });
    }
}

fn drawMnistDigit(pixels: []const f32) void {
    for (0..28) |y| {
        for (0..28) |x| {
            const num = pixels[x + y * 28];
            const col: u32 = 232 + @as(u32, @intFromFloat(num * 23.0));
            std.debug.print("\x1b[48;5;{d}m  ", .{col});
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("\x1b[0m", .{});
}

fn matLoad(allocator: std.mem.Allocator, filename: []const u8, rows: u32, cols: u32) !Matrix {
    var mat = try Matrix.init(allocator, rows, cols);
    errdefer mat.deinit();

    const file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();

    _ = try file.readAll(std.mem.sliceAsBytes(mat.data));
    return mat;
}

fn createMnistModel(allocator: std.mem.Allocator, model: *ModelContext, rng: std.Random) !void {
    const input = try mvCreate(allocator, model, 784, 1, MV.input);

    const w0 = try mvCreate(allocator, model, 16, 784, MV.requires_grad | MV.parameter);
    const w1 = try mvCreate(allocator, model, 16, 16, MV.requires_grad | MV.parameter);
    const w2 = try mvCreate(allocator, model, 10, 16, MV.requires_grad | MV.parameter);

    const bound0: f32 = @sqrt(6.0 / (784.0 + 16.0));
    const bound1: f32 = @sqrt(6.0 / (16.0 + 16.0));
    const bound2: f32 = @sqrt(6.0 / (16.0 + 10.0));
    for (w0.val.data) |*v| v.* = rng.float(f32) * (2.0 * bound0) - bound0;
    for (w1.val.data) |*v| v.* = rng.float(f32) * (2.0 * bound1) - bound1;
    for (w2.val.data) |*v| v.* = rng.float(f32) * (2.0 * bound2) - bound2;

    const b0 = try mvCreate(allocator, model, 16, 1, MV.requires_grad | MV.parameter);
    const b1 = try mvCreate(allocator, model, 16, 1, MV.requires_grad | MV.parameter);
    const b2 = try mvCreate(allocator, model, 10, 1, MV.requires_grad | MV.parameter);

    const z0_a = try mvMatmul(allocator, model, w0, input, MV.none);
    const z0_b = try mvAdd(allocator, model, z0_a, b0, MV.none);
    const a0 = try mvRelu(allocator, model, z0_b, MV.none);

    const z1_a = try mvMatmul(allocator, model, w1, a0, MV.none);
    const z1_b = try mvAdd(allocator, model, z1_a, b1, MV.none);
    const z1_c = try mvRelu(allocator, model, z1_b, MV.none);
    const a1 = try mvAdd(allocator, model, a0, z1_c, MV.none);

    const z2_a = try mvMatmul(allocator, model, w2, a1, MV.none);
    const z2_b = try mvAdd(allocator, model, z2_a, b2, MV.none);
    _ = try mvSoftmax(allocator, model, z2_b, MV.output);

    const y = try mvCreate(allocator, model, 10, 1, MV.desired_output);
    _ = try mvCrossEntropy(allocator, model, y, model.output.?, MV.cost);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();

    var prng = std.Random.DefaultPrng.init(0x1234_5678);
    const rng = prng.random();

    var train_images = try matLoad(allocator, "train_images.bin", 60000, 784);
    var test_images = try matLoad(allocator, "test_images.bin", 10000, 784);
    const train_label_ids = try matLoad(allocator, "train_labels.bin", 60000, 1);
    const test_label_ids = try matLoad(allocator, "test_labels.bin", 10000, 1);

    var train_labels = try Matrix.init(allocator, 60000, 10);
    var test_labels = try Matrix.init(allocator, 10000, 10);

    for (0..60000) |i| {
        const num: usize = @intFromFloat(train_label_ids.data[i]);
        train_labels.data[i * 10 + num] = 1.0;
    }
    for (0..10000) |i| {
        const num: usize = @intFromFloat(test_label_ids.data[i]);
        test_labels.data[i * 10 + num] = 1.0;
    }

    drawMnistDigit(test_images.data[0..784]);

    var model: ModelContext = .{};
    try createMnistModel(allocator, &model, rng);
    try modelCompile(allocator, &model);

    @memcpy(model.input.?.val.data[0..784], test_images.data[0..784]);
    try modelFeedForward(&model);
    std.debug.print("pre-training output:\n", .{});
    for (0..10) |i| std.debug.print("{d:.3} ", .{model.output.?.val.data[i]});
    std.debug.print("\n", .{});

    const training_desc: ModelTrainingDesc = .{
        .train_images = &train_images,
        .train_labels = &train_labels,
        .test_images = &test_images,
        .test_labels = &test_labels,
        .epochs = 2,
        .batch_size = 50,
        .learning_rate = 0.01,
    };
    try modelTrain(&model, &training_desc, allocator, rng);

    @memcpy(model.input.?.val.data[0..784], test_images.data[0..784]);
    try modelFeedForward(&model);
    std.debug.print("post-training output:\n", .{});
    for (0..10) |i| std.debug.print("{d:.3} ", .{model.output.?.val.data[i]});
    std.debug.print("\n", .{});
}

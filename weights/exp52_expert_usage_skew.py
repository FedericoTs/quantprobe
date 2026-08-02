"""exp52_expert_usage_skew.py -- pre-registration #87 (task #52).

Measures per-expert routing concentration on our two MoE rows over a FIXED, hash-pinned
prompt set (.NET/C# code + WikiText-2 prose), and scores it against the kill rule staked in
preregistrations/2026-07-30-expert-usage-skew.md BEFORE any of this was run.

THE INSTRUMENT.  llama.cpp's `llama-imatrix` already counts, per MoE layer and per expert, how
many (token, slot) pairs the router sent to that expert: `e.counts[ex]++` in
tools/imatrix/imatrix.cpp under GGML_OP_MUL_MAT_ID.  It writes those counts into the output
imatrix GGUF as an F32 tensor `blk.<N>.ffn_<gate|up|down>_exps.weight.counts` of length
n_expert.  So expert-selection frequency is obtainable from a STOCK, ALREADY-BUILT binary with
no patch and no custom runtime.  This script drives that binary and reads those tensors.

WHY THIS IS NOT A TIMING MEASUREMENT (AND WHERE C-14 STILL APPLIES).  We record routing COUNTS,
not tok/s.  Counts do not depend on clocks, thermals or placement, so the TIMING form of C-14
(one cal_id per comparison) does not bind.  They DO depend on the binary, the model file, the
corpus, the chunk budget, and -- through CPU float reduction order at router near-ties -- on
thread count and offload split.  So every imatrix carries a run stamp (<file>.stamp.json) and a
cached file is reused only if the stamp matches exactly; otherwise the script refuses.  That
closes the C-14-shaped hole where shard A and shard B of one comparison could be measured under
different settings.  Runs default to -ngl 0 (pure CPU) for determinism and to sidestep the 6 GB
VRAM limit on a 30B file; --ngl overrides (and demotes the run to SWEEP).

CONTROLS.  `--selftest` drives four synthetic routing distributions with known ground truth
through this exact analysis path: flat (must FAIL G1 at the structural null), domain-disjoint
(must FAIL), partial-overlap (G1 must PASS while G2 FAILS -- the proof that the stability gate
can overturn a strong headline on its own), and shared-hot (must PASS -- the proof that a
positive result is reachable at all).  The measured run executes them first and refuses to emit
any verdict if they do not reproduce.

REFUSES RATHER THAN ESTIMATES.  Missing chunk metadata, a zero-mass layer, mismatched layer sets
or model geometry across shards, a single domain (which would make the retention gate pass
vacuously), an unpinned corpus shard, or a missing staked binary all raise Refuse.  None of them
fall back to a default -- a silently substituted default here would publish an artifact as a
measurement.

DESIGN: EVERY HEADLINE NUMBER IS HELD OUT.  Ranking experts and scoring the same counts is a
winner's-curse estimate -- the top-k of a noisy sample always looks hotter than it is.  So each
(model, domain) is measured as TWO independent shards A and B.  Hot sets are always chosen on
shard A and scored on shard B.  The in-sample number is still reported, but only as the
inflated bound it is.

Outputs (all under weights/data/):
  exp52_imatrix/<model>__<domain>__<shard>.imatrix.gguf   raw per-run imatrix
  exp52_corpus_code_<A|B>.txt, exp52_corpus_prose_<A|B>.txt   the pinned corpus shards
  exp52_expert_usage_skew.json                             machine-readable results
  exp52_expert_usage_skew.log                              human-readable transcript

Idempotent: an imatrix run whose output already exists and validates is skipped (--force
redoes them).  Analysis is always recomputed from the raw files.

Run:
  python weights/exp52_expert_usage_skew.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

# --------------------------------------------------------------------------------------
# STAKED CONSTANTS -- these mirror preregistration #87 and must not be edited after staking.
# --------------------------------------------------------------------------------------

PREREG = "preregistrations/2026-07-30-expert-usage-skew.md"
PREREG_ID = 87

# Residency budget under test: the fraction of experts we are allowed to pin.  0.32 matches the
# expert-layer share the shipped -ot regex keeps resident (U-14 / prereg #62), so the frequency
# -ranked pin and the shipped index-ranked pin cost the SAME VRAM.
FRAC = 0.32

# G1: held-out routing mass carried by the pooled top-FRAC hot set, on the WORST domain.
# Break-even derivation is in the prereg (1.5x cut in the streamed-expert byte term).
GATE_G1_MIN_SHARE = 0.55

# G2: cross-domain retention.  A static pin is chosen once; if the set fitted on code loses
# more than 10% of its mass on prose (or vice versa), static pinning is refuted even if raw
# concentration is high.
GATE_G2_MIN_RETENTION = 0.90

# The structural null: an index-chosen 32% of experts carries 32% of routing mass.
BASELINE_INDEX_SHARE = FRAC

# If the headline lands within this of the gate, the run is declared MARGINAL, not PASS/FAIL.
# NOTE: marginality applies to G1 ONLY. A decisive G2 (retention) failure is never softened into
# MARGINAL -- see verdict_for(). Otherwise KR-2 would be unable to fire in the 0.53-0.57 band.
MARGIN_BAND = 0.02

# STAKED MEASUREMENT BUDGET.  These are not free parameters: the headline verdict is defined at
# this budget and no other.  The CLI can override them, but doing so demotes the run to SWEEP so
# it can never be read back as the staked result (KR-4).
STAKED_CTX = 512
STAKED_CHUNKS = 8
STAKED_MIN_CHUNKS = 8
STAKED_NGL = 0

# Corpus shard hashes.  Pinned in the prereg; the script refuses to analyse anything else.
CORPUS_SHA256 = {
    "code_A": "7fe9821f1570dabdae866a4ad19eceead74475b831028813c0b2a03f28429bd3",
    "code_B": "dccbb1bd4af5b47b363d6cce7090bbebd73c9cf112249583f92a98618f5a1593",
    "prose_A": "09e3648bef826e9ca14744249aaf3cbb5c0a5455efdde7a9138663fbfc9dea42",
    "prose_B": "5fb4087ab6b5c048366614e0525127f3adb70055eb1efeead2e9eb9ac785d304",
}

MODELS = {
    "qwen3-30b-a3b-q2k": r"D:\evo-compress-data\gguf\Qwen3-30B-A3B-Q2_K.gguf",
    "q35-A-shexp": r"D:\evo-compress-data\gguf\q35-A-shexp.gguf",
}

PROSE_SRC = r"D:\evo-compress-data\eval\wiki.test.raw"
PROSE_SRC_SHA256 = "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"
PROSE_SHARD_BYTES = 120_000

DEFAULT_IMATRIX_BIN = r"<repo>\tools\llamacpp-b10098\llama-imatrix.exe"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "weights", "data")
IMAT_DIR = os.path.join(DATA, "exp52_imatrix")

COUNTS_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight\.counts$")


class Refuse(Exception):
    """A precondition is missing.  We stop loudly rather than emit a wrong number."""


# --------------------------------------------------------------------------------------
# The .NET/C# corpus.  Routing is input-dependent, so a code-only or prose-only sample would
# not generalise; this is the code half.  Hand-authored for this experiment (no C# source
# existed on the box), deliberately varied across ASP.NET Core, EF Core, LINQ, async, spans,
# channels, generics, source-gen attributes, and xUnit.  Pinned by SHA-256 in the prereg.
# --------------------------------------------------------------------------------------

CODE_CORPUS = r'''
// ============================ Domain/Orders.cs ============================
using System;
using System.Collections.Generic;
using System.Linq;

namespace Contoso.Ordering.Domain;

public readonly record struct Money(decimal Amount, string Currency)
{
    public static Money Zero(string currency) => new(0m, currency);

    public Money Add(Money other)
    {
        if (!string.Equals(Currency, other.Currency, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException($"Cannot add {Currency} to {other.Currency}.");
        }
        return this with { Amount = Amount + other.Amount };
    }

    public override string ToString() => $"{Amount:N2} {Currency}";
}

public enum OrderStatus
{
    Draft = 0,
    Submitted = 1,
    Paid = 2,
    Shipped = 3,
    Cancelled = 4
}

public sealed class OrderLine
{
    public Guid Id { get; private set; } = Guid.NewGuid();
    public string Sku { get; private set; }
    public int Quantity { get; private set; }
    public Money UnitPrice { get; private set; }

    private OrderLine() { Sku = string.Empty; }

    public OrderLine(string sku, int quantity, Money unitPrice)
    {
        if (string.IsNullOrWhiteSpace(sku))
        {
            throw new ArgumentException("SKU is required.", nameof(sku));
        }
        if (quantity <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(quantity), quantity, "Quantity must be positive.");
        }
        Sku = sku;
        Quantity = quantity;
        UnitPrice = unitPrice;
    }

    public Money LineTotal => UnitPrice with { Amount = UnitPrice.Amount * Quantity };

    public void Restock(int delta)
    {
        var next = Quantity + delta;
        if (next < 0)
        {
            throw new InvalidOperationException("Cannot restock below zero.");
        }
        Quantity = next;
    }
}

public sealed class Order
{
    private readonly List<OrderLine> _lines = new();

    public Guid Id { get; private set; }
    public string CustomerId { get; private set; }
    public OrderStatus Status { get; private set; }
    public DateTimeOffset CreatedUtc { get; private set; }
    public DateTimeOffset? SubmittedUtc { get; private set; }
    public IReadOnlyList<OrderLine> Lines => _lines.AsReadOnly();

    private Order()
    {
        CustomerId = string.Empty;
    }

    public Order(string customerId, TimeProvider? clock = null)
    {
        Id = Guid.NewGuid();
        CustomerId = customerId ?? throw new ArgumentNullException(nameof(customerId));
        Status = OrderStatus.Draft;
        CreatedUtc = (clock ?? TimeProvider.System).GetUtcNow();
    }

    public void AddLine(string sku, int quantity, Money unitPrice)
    {
        EnsureDraft();
        var existing = _lines.FirstOrDefault(l => l.Sku == sku);
        if (existing is not null)
        {
            existing.Restock(quantity);
            return;
        }
        _lines.Add(new OrderLine(sku, quantity, unitPrice));
    }

    public void RemoveLine(string sku)
    {
        EnsureDraft();
        _lines.RemoveAll(l => l.Sku == sku);
    }

    public Money Total(string currency) =>
        _lines.Aggregate(Money.Zero(currency), (acc, line) => acc.Add(line.LineTotal));

    public void Submit(TimeProvider? clock = null)
    {
        EnsureDraft();
        if (_lines.Count == 0)
        {
            throw new InvalidOperationException("Cannot submit an empty order.");
        }
        Status = OrderStatus.Submitted;
        SubmittedUtc = (clock ?? TimeProvider.System).GetUtcNow();
    }

    public void Cancel(string reason)
    {
        if (Status is OrderStatus.Shipped)
        {
            throw new InvalidOperationException($"Order {Id} already shipped; reason='{reason}'.");
        }
        Status = OrderStatus.Cancelled;
    }

    private void EnsureDraft()
    {
        if (Status != OrderStatus.Draft)
        {
            throw new InvalidOperationException($"Order {Id} is {Status}, not Draft.");
        }
    }
}

// ============================ Domain/Result.cs ============================
namespace Contoso.Ordering.Domain;

public readonly struct Error
{
    public string Code { get; }
    public string Message { get; }

    public Error(string code, string message)
    {
        Code = code;
        Message = message;
    }

    public static readonly Error None = new(string.Empty, string.Empty);
    public bool IsNone => string.IsNullOrEmpty(Code);
}

public readonly struct Result<T>
{
    private readonly T? _value;

    private Result(T value)
    {
        _value = value;
        Error = Error.None;
        IsSuccess = true;
    }

    private Result(Error error)
    {
        _value = default;
        Error = error;
        IsSuccess = false;
    }

    public bool IsSuccess { get; }
    public bool IsFailure => !IsSuccess;
    public Error Error { get; }

    public T Value => IsSuccess
        ? _value!
        : throw new InvalidOperationException($"Result is a failure: {Error.Code}");

    public static Result<T> Success(T value) => new(value);
    public static Result<T> Failure(Error error) => new(error);

    public Result<TOut> Map<TOut>(Func<T, TOut> selector) =>
        IsSuccess ? Result<TOut>.Success(selector(Value)) : Result<TOut>.Failure(Error);

    public Result<TOut> Bind<TOut>(Func<T, Result<TOut>> binder) =>
        IsSuccess ? binder(Value) : Result<TOut>.Failure(Error);

    public TOut Match<TOut>(Func<T, TOut> onSuccess, Func<Error, TOut> onFailure) =>
        IsSuccess ? onSuccess(Value) : onFailure(Error);
}

// ============================ Infrastructure/OrderingDbContext.cs ============================
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using Contoso.Ordering.Domain;

namespace Contoso.Ordering.Infrastructure;

public sealed class OrderingDbContext : DbContext
{
    public OrderingDbContext(DbContextOptions<OrderingDbContext> options) : base(options)
    {
    }

    public DbSet<Order> Orders => Set<Order>();
    public DbSet<OrderLine> OrderLines => Set<OrderLine>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.ApplyConfigurationsFromAssembly(typeof(OrderingDbContext).Assembly);
        modelBuilder.HasDefaultSchema("ordering");
        base.OnModelCreating(modelBuilder);
    }

    protected override void ConfigureConventions(ModelConfigurationBuilder builder)
    {
        builder.Properties<decimal>().HavePrecision(18, 4);
        builder.Properties<string>().HaveMaxLength(512);
    }
}

public sealed class OrderConfiguration : IEntityTypeConfiguration<Order>
{
    public void Configure(EntityTypeBuilder<Order> builder)
    {
        builder.ToTable("orders");
        builder.HasKey(o => o.Id);
        builder.Property(o => o.CustomerId).IsRequired().HasMaxLength(64);
        builder.Property(o => o.Status).HasConversion<string>().HasMaxLength(32);
        builder.Property(o => o.CreatedUtc).HasColumnName("created_utc");
        builder.HasIndex(o => new { o.CustomerId, o.Status }).HasDatabaseName("ix_orders_customer_status");
        builder.OwnsMany(o => o.Lines, lines =>
        {
            lines.ToTable("order_lines");
            lines.WithOwner().HasForeignKey("OrderId");
            lines.HasKey(l => l.Id);
            lines.Property(l => l.Sku).IsRequired().HasMaxLength(64);
            lines.OwnsOne(l => l.UnitPrice, price =>
            {
                price.Property(p => p.Amount).HasColumnName("unit_price_amount");
                price.Property(p => p.Currency).HasColumnName("unit_price_currency").HasMaxLength(3);
            });
        });
        builder.Navigation(o => o.Lines).UsePropertyAccessMode(PropertyAccessMode.Field);
    }
}

// ============================ Infrastructure/OrderRepository.cs ============================
using System.Threading;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;

namespace Contoso.Ordering.Infrastructure;

public interface IOrderRepository
{
    Task<Order?> FindAsync(Guid id, CancellationToken cancellationToken = default);
    Task<IReadOnlyList<Order>> ListByCustomerAsync(string customerId, CancellationToken cancellationToken = default);
    Task AddAsync(Order order, CancellationToken cancellationToken = default);
    Task<int> SaveChangesAsync(CancellationToken cancellationToken = default);
}

public sealed class OrderRepository : IOrderRepository
{
    private readonly OrderingDbContext _db;
    private readonly ILogger<OrderRepository> _logger;

    public OrderRepository(OrderingDbContext db, ILogger<OrderRepository> logger)
    {
        _db = db;
        _logger = logger;
    }

    public async Task<Order?> FindAsync(Guid id, CancellationToken cancellationToken = default)
    {
        return await _db.Orders
            .Include(o => o.Lines)
            .AsSplitQuery()
            .FirstOrDefaultAsync(o => o.Id == id, cancellationToken)
            .ConfigureAwait(false);
    }

    public async Task<IReadOnlyList<Order>> ListByCustomerAsync(string customerId, CancellationToken cancellationToken = default)
    {
        var orders = await _db.Orders
            .AsNoTracking()
            .Where(o => o.CustomerId == customerId && o.Status != OrderStatus.Cancelled)
            .OrderByDescending(o => o.CreatedUtc)
            .Take(200)
            .ToListAsync(cancellationToken)
            .ConfigureAwait(false);

        _logger.LogDebug("Loaded {Count} orders for {CustomerId}", orders.Count, customerId);
        return orders;
    }

    public async Task AddAsync(Order order, CancellationToken cancellationToken = default)
    {
        await _db.Orders.AddAsync(order, cancellationToken).ConfigureAwait(false);
    }

    public Task<int> SaveChangesAsync(CancellationToken cancellationToken = default) =>
        _db.SaveChangesAsync(cancellationToken);
}

// ============================ Application/OrderService.cs ============================
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace Contoso.Ordering.Application;

public sealed class OrderingOptions
{
    public const string SectionName = "Ordering";

    public int MaxLinesPerOrder { get; set; } = 100;
    public string DefaultCurrency { get; set; } = "EUR";
    public TimeSpan SubmitTimeout { get; set; } = TimeSpan.FromSeconds(30);
    public bool EnableFraudScreening { get; set; } = true;
}

public sealed record SubmitOrderCommand(Guid OrderId, string IdempotencyKey);

public sealed record OrderSummaryDto(
    Guid Id,
    string CustomerId,
    string Status,
    decimal Total,
    string Currency,
    int LineCount);

public interface IOrderService
{
    Task<Result<OrderSummaryDto>> SubmitAsync(SubmitOrderCommand command, CancellationToken cancellationToken);
}

public sealed class OrderService : IOrderService
{
    private static readonly ActivitySource ActivitySource = new("Contoso.Ordering");

    private readonly IOrderRepository _repository;
    private readonly IFraudScreener _fraud;
    private readonly OrderingOptions _options;
    private readonly ILogger<OrderService> _logger;

    public OrderService(
        IOrderRepository repository,
        IFraudScreener fraud,
        IOptions<OrderingOptions> options,
        ILogger<OrderService> logger)
    {
        _repository = repository;
        _fraud = fraud;
        _options = options.Value;
        _logger = logger;
    }

    public async Task<Result<OrderSummaryDto>> SubmitAsync(
        SubmitOrderCommand command,
        CancellationToken cancellationToken)
    {
        using var activity = ActivitySource.StartActivity("SubmitOrder", ActivityKind.Internal);
        activity?.SetTag("order.id", command.OrderId);
        activity?.SetTag("idempotency.key", command.IdempotencyKey);

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutCts.CancelAfter(_options.SubmitTimeout);

        var order = await _repository.FindAsync(command.OrderId, timeoutCts.Token).ConfigureAwait(false);
        if (order is null)
        {
            _logger.LogWarning("Order {OrderId} not found", command.OrderId);
            return Result<OrderSummaryDto>.Failure(new Error("order.not_found", "Order does not exist."));
        }

        if (order.Lines.Count > _options.MaxLinesPerOrder)
        {
            return Result<OrderSummaryDto>.Failure(
                new Error("order.too_many_lines", $"Order has {order.Lines.Count} lines; max is {_options.MaxLinesPerOrder}."));
        }

        if (_options.EnableFraudScreening)
        {
            var verdict = await _fraud.ScreenAsync(order, timeoutCts.Token).ConfigureAwait(false);
            if (verdict.IsFailure)
            {
                _logger.LogWarning("Fraud screening rejected {OrderId}: {Code}", order.Id, verdict.Error.Code);
                return Result<OrderSummaryDto>.Failure(verdict.Error);
            }
        }

        try
        {
            order.Submit();
            await _repository.SaveChangesAsync(timeoutCts.Token).ConfigureAwait(false);
        }
        catch (DbUpdateConcurrencyException ex)
        {
            _logger.LogError(ex, "Concurrency conflict submitting {OrderId}", order.Id);
            return Result<OrderSummaryDto>.Failure(new Error("order.conflict", "Order was modified concurrently."));
        }
        catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
        {
            return Result<OrderSummaryDto>.Failure(new Error("order.timeout", "Submit exceeded the configured timeout."));
        }

        var total = order.Total(_options.DefaultCurrency);
        return Result<OrderSummaryDto>.Success(new OrderSummaryDto(
            order.Id,
            order.CustomerId,
            order.Status.ToString(),
            total.Amount,
            total.Currency,
            order.Lines.Count));
    }
}

// ============================ Api/OrdersController.cs ============================
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace Contoso.Ordering.Api;

[ApiController]
[Route("api/v{version:apiVersion}/orders")]
[Authorize(Policy = "OrdersWrite")]
[Produces("application/json")]
public sealed class OrdersController : ControllerBase
{
    private readonly IOrderService _orders;
    private readonly IOrderRepository _repository;

    public OrdersController(IOrderService orders, IOrderRepository repository)
    {
        _orders = orders;
        _repository = repository;
    }

    [HttpGet("{id:guid}")]
    [ProducesResponseType(typeof(OrderSummaryDto), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    [ResponseCache(Duration = 15, Location = ResponseCacheLocation.Private)]
    public async Task<ActionResult<OrderSummaryDto>> GetById(Guid id, CancellationToken cancellationToken)
    {
        var order = await _repository.FindAsync(id, cancellationToken);
        if (order is null)
        {
            return NotFound(new ProblemDetails
            {
                Title = "Order not found",
                Status = StatusCodes.Status404NotFound,
                Detail = $"No order with id {id}."
            });
        }

        var total = order.Total("EUR");
        return Ok(new OrderSummaryDto(order.Id, order.CustomerId, order.Status.ToString(), total.Amount, total.Currency, order.Lines.Count));
    }

    [HttpPost("{id:guid}/submit")]
    [ProducesResponseType(typeof(OrderSummaryDto), StatusCodes.Status202Accepted)]
    [ProducesResponseType(typeof(ProblemDetails), StatusCodes.Status409Conflict)]
    public async Task<IActionResult> Submit(
        Guid id,
        [FromHeader(Name = "Idempotency-Key")] string idempotencyKey,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(idempotencyKey))
        {
            ModelState.AddModelError("Idempotency-Key", "Header is required.");
            return ValidationProblem(ModelState);
        }

        var result = await _orders.SubmitAsync(new SubmitOrderCommand(id, idempotencyKey), cancellationToken);

        return result.Match<IActionResult>(
            onSuccess: dto => AcceptedAtAction(nameof(GetById), new { id = dto.Id }, dto),
            onFailure: error => error.Code switch
            {
                "order.not_found" => NotFound(new ProblemDetails { Title = error.Message, Status = 404 }),
                "order.conflict" => Conflict(new ProblemDetails { Title = error.Message, Status = 409 }),
                "order.timeout" => StatusCode(StatusCodes.Status504GatewayTimeout),
                _ => BadRequest(new ProblemDetails { Title = error.Message, Status = 400 })
            });
    }
}

// ============================ Api/Program.cs ============================
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Diagnostics.HealthChecks;
using Serilog;

var builder = WebApplication.CreateBuilder(args);

builder.Host.UseSerilog((context, services, configuration) => configuration
    .ReadFrom.Configuration(context.Configuration)
    .ReadFrom.Services(services)
    .Enrich.FromLogContext()
    .WriteTo.Console(outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj} {Properties:j}{NewLine}{Exception}"));

builder.Services.Configure<OrderingOptions>(builder.Configuration.GetSection(OrderingOptions.SectionName));

builder.Services.AddDbContextPool<OrderingDbContext>((sp, options) =>
{
    var connectionString = builder.Configuration.GetConnectionString("Ordering")
        ?? throw new InvalidOperationException("Connection string 'Ordering' is missing.");
    options.UseNpgsql(connectionString, npgsql =>
    {
        npgsql.EnableRetryOnFailure(maxRetryCount: 5, maxRetryDelay: TimeSpan.FromSeconds(10), errorCodesToAdd: null);
        npgsql.MigrationsHistoryTable("__ef_migrations", "ordering");
    });
    options.UseSnakeCaseNamingConvention();
});

builder.Services.AddScoped<IOrderRepository, OrderRepository>();
builder.Services.AddScoped<IOrderService, OrderService>();
builder.Services.AddSingleton<IFraudScreener, HeuristicFraudScreener>();
builder.Services.AddHostedService<OutboxDispatcher>();

builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
        options.JsonSerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
        options.JsonSerializerOptions.Converters.Add(new JsonStringEnumConverter());
        options.JsonSerializerOptions.NumberHandling = JsonNumberHandling.AllowReadingFromString;
    });

builder.Services.AddHealthChecks()
    .AddDbContextCheck<OrderingDbContext>("ef-core")
    .AddCheck<OutboxLagHealthCheck>("outbox-lag", tags: new[] { "ready" });

builder.Services.AddRateLimiter(options =>
{
    options.AddFixedWindowLimiter("api", limiter =>
    {
        limiter.PermitLimit = 200;
        limiter.Window = TimeSpan.FromMinutes(1);
        limiter.QueueProcessingOrder = QueueProcessingOrder.OldestFirst;
        limiter.QueueLimit = 20;
    });
});

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseDeveloperExceptionPage();
}
else
{
    app.UseExceptionHandler("/error");
    app.UseHsts();
}

app.UseSerilogRequestLogging();
app.UseHttpsRedirection();
app.UseRateLimiter();
app.UseAuthentication();
app.UseAuthorization();
app.MapControllers().RequireRateLimiting("api");

app.MapHealthChecks("/health/ready", new HealthCheckOptions
{
    Predicate = registration => registration.Tags.Contains("ready")
});

app.MapGet("/api/v1/orders/{id:guid}/lines", async (Guid id, IOrderRepository repo, CancellationToken ct) =>
{
    var order = await repo.FindAsync(id, ct);
    return order is null
        ? Results.NotFound()
        : Results.Ok(order.Lines.Select(l => new { l.Sku, l.Quantity, Total = l.LineTotal.Amount }));
})
.WithName("GetOrderLines")
.WithOpenApi();

await app.RunAsync();

// ============================ Infrastructure/OutboxDispatcher.cs ============================
using System.Threading.Channels;
using Microsoft.Extensions.Hosting;

namespace Contoso.Ordering.Infrastructure;

public sealed record OutboxMessage(long Sequence, string Type, string Payload, DateTimeOffset EnqueuedUtc);

public sealed class OutboxDispatcher : BackgroundService
{
    private readonly Channel<OutboxMessage> _channel;
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly ILogger<OutboxDispatcher> _logger;
    private readonly PeriodicTimer _timer;

    public OutboxDispatcher(IServiceScopeFactory scopeFactory, ILogger<OutboxDispatcher> logger)
    {
        _scopeFactory = scopeFactory;
        _logger = logger;
        _timer = new PeriodicTimer(TimeSpan.FromSeconds(2));
        _channel = Channel.CreateBounded<OutboxMessage>(new BoundedChannelOptions(2048)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = false,
            SingleWriter = true
        });
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var producer = ProduceAsync(stoppingToken);
        var consumers = Enumerable
            .Range(0, Environment.ProcessorCount / 2)
            .Select(_ => ConsumeAsync(stoppingToken))
            .ToArray();

        await Task.WhenAll(consumers.Append(producer)).ConfigureAwait(false);
    }

    private async Task ProduceAsync(CancellationToken stoppingToken)
    {
        while (await _timer.WaitForNextTickAsync(stoppingToken).ConfigureAwait(false))
        {
            using var scope = _scopeFactory.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<OrderingDbContext>();

            var batch = await db.Database
                .SqlQuery<OutboxMessage>($"SELECT sequence, type, payload, enqueued_utc FROM ordering.outbox WHERE dispatched_utc IS NULL ORDER BY sequence LIMIT 256")
                .ToListAsync(stoppingToken)
                .ConfigureAwait(false);

            foreach (var message in batch)
            {
                await _channel.Writer.WriteAsync(message, stoppingToken).ConfigureAwait(false);
            }
        }
        _channel.Writer.TryComplete();
    }

    private async Task ConsumeAsync(CancellationToken stoppingToken)
    {
        await foreach (var message in _channel.Reader.ReadAllAsync(stoppingToken).ConfigureAwait(false))
        {
            try
            {
                await PublishAsync(message, stoppingToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.LogError(ex, "Failed to dispatch outbox message {Sequence} of type {Type}", message.Sequence, message.Type);
            }
        }
    }

    private static Task PublishAsync(OutboxMessage message, CancellationToken cancellationToken) =>
        Task.CompletedTask;

    public override void Dispose()
    {
        _timer.Dispose();
        base.Dispose();
    }
}

// ============================ Infrastructure/Fraud.cs ============================
namespace Contoso.Ordering.Infrastructure;

public interface IFraudScreener
{
    ValueTask<Result<bool>> ScreenAsync(Order order, CancellationToken cancellationToken);
}

public sealed partial class HeuristicFraudScreener : IFraudScreener
{
    private static readonly HashSet<string> BlockedSkuPrefixes = new(StringComparer.OrdinalIgnoreCase)
    {
        "GIFTCARD-", "CRYPTO-", "PREPAID-"
    };

    [GeneratedRegex(@"^[A-Z]{2,4}-\d{4,8}$", RegexOptions.CultureInvariant)]
    private static partial Regex SkuPattern();

    public ValueTask<Result<bool>> ScreenAsync(Order order, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        foreach (var line in order.Lines)
        {
            if (!SkuPattern().IsMatch(line.Sku))
            {
                return ValueTask.FromResult(Result<bool>.Failure(new Error("fraud.bad_sku", $"Malformed SKU '{line.Sku}'.")));
            }

            if (BlockedSkuPrefixes.Any(prefix => line.Sku.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)))
            {
                return ValueTask.FromResult(Result<bool>.Failure(new Error("fraud.blocked_sku", "SKU category requires manual review.")));
            }
        }

        var total = order.Total("EUR").Amount;
        var risk = total switch
        {
            < 100m => 0.01,
            >= 100m and < 1_000m => 0.05,
            >= 1_000m and < 10_000m => 0.25,
            _ => 0.80
        };

        return ValueTask.FromResult(risk < 0.5
            ? Result<bool>.Success(true)
            : Result<bool>.Failure(new Error("fraud.high_risk", $"Risk score {risk:P0} exceeds threshold.")));
    }
}

// ============================ Infrastructure/Buffers.cs ============================
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Contoso.Ordering.Infrastructure;

public static class SkuCodec
{
    private const int MaxStackAlloc = 256;

    public static int Encode(ReadOnlySpan<char> sku, Span<byte> destination)
    {
        if (destination.Length < sku.Length)
        {
            throw new ArgumentException("Destination too small.", nameof(destination));
        }

        var written = 0;
        for (var i = 0; i < sku.Length; i++)
        {
            var c = sku[i];
            if (c == '-')
            {
                continue;
            }
            destination[written++] = (byte)char.ToUpperInvariant(c);
        }
        return written;
    }

    public static string Normalize(string sku)
    {
        var length = sku.Length;
        byte[]? rented = null;
        var buffer = length <= MaxStackAlloc
            ? stackalloc byte[MaxStackAlloc]
            : (rented = ArrayPool<byte>.Shared.Rent(length));

        try
        {
            var written = Encode(sku, buffer);
            return System.Text.Encoding.ASCII.GetString(buffer[..written]);
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<byte>.Shared.Return(rented, clearArray: true);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong FingerprintOf(ReadOnlySpan<byte> data)
    {
        const ulong offset = 14695981039346656037UL;
        const ulong prime = 1099511628211UL;

        var hash = offset;
        ref var start = ref MemoryMarshal.GetReference(data);
        for (var i = 0; i < data.Length; i++)
        {
            hash ^= Unsafe.Add(ref start, i);
            hash *= prime;
        }
        return hash;
    }
}

// ============================ Api/Middleware.cs ============================
namespace Contoso.Ordering.Api;

public sealed class CorrelationIdMiddleware
{
    private const string HeaderName = "X-Correlation-Id";

    private readonly RequestDelegate _next;
    private readonly ILogger<CorrelationIdMiddleware> _logger;

    public CorrelationIdMiddleware(RequestDelegate next, ILogger<CorrelationIdMiddleware> logger)
    {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        if (!context.Request.Headers.TryGetValue(HeaderName, out var correlationId) || StringValues.IsNullOrEmpty(correlationId))
        {
            correlationId = Activity.Current?.TraceId.ToString() ?? Guid.NewGuid().ToString("N");
        }

        context.Response.OnStarting(() =>
        {
            context.Response.Headers[HeaderName] = correlationId;
            return Task.CompletedTask;
        });

        using (_logger.BeginScope(new Dictionary<string, object> { ["CorrelationId"] = correlationId.ToString() }))
        {
            await _next(context).ConfigureAwait(false);
        }
    }
}

public static class MiddlewareExtensions
{
    public static IApplicationBuilder UseCorrelationId(this IApplicationBuilder app) =>
        app.UseMiddleware<CorrelationIdMiddleware>();

    public static IServiceCollection AddOrderingCore(this IServiceCollection services, IConfiguration configuration)
    {
        services.AddOptions<OrderingOptions>()
            .Bind(configuration.GetSection(OrderingOptions.SectionName))
            .ValidateDataAnnotations()
            .Validate(options => options.MaxLinesPerOrder > 0, "MaxLinesPerOrder must be positive.")
            .ValidateOnStart();

        return services;
    }
}

// ============================ Application/Queries.cs ============================
namespace Contoso.Ordering.Application;

public static class OrderQueries
{
    public static IQueryable<OrderSummaryDto> ProjectSummaries(this IQueryable<Order> source, string currency) =>
        source.Select(o => new OrderSummaryDto(
            o.Id,
            o.CustomerId,
            o.Status.ToString(),
            o.Lines.Sum(l => l.UnitPrice.Amount * l.Quantity),
            currency,
            o.Lines.Count));

    public static IEnumerable<IGrouping<string, OrderSummaryDto>> GroupByCustomer(IEnumerable<OrderSummaryDto> summaries) =>
        summaries
            .Where(s => s.Total > 0)
            .OrderBy(s => s.CustomerId)
            .ThenByDescending(s => s.Total)
            .GroupBy(s => s.CustomerId);

    public static Dictionary<string, decimal> RevenueByStatus(IEnumerable<OrderSummaryDto> summaries) =>
        summaries
            .GroupBy(s => s.Status)
            .ToDictionary(g => g.Key, g => g.Sum(s => s.Total));

    public static async IAsyncEnumerable<OrderSummaryDto> StreamAsync(
        IQueryable<Order> source,
        string currency,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var order in source.AsAsyncEnumerable().WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            var total = order.Total(currency);
            yield return new OrderSummaryDto(order.Id, order.CustomerId, order.Status.ToString(), total.Amount, total.Currency, order.Lines.Count);
        }
    }
}

// ============================ Tests/OrderServiceTests.cs ============================
using FluentAssertions;
using Moq;
using Xunit;

namespace Contoso.Ordering.Tests;

public sealed class OrderServiceTests
{
    private readonly Mock<IOrderRepository> _repository = new(MockBehavior.Strict);
    private readonly Mock<IFraudScreener> _fraud = new();
    private readonly OrderService _sut;

    public OrderServiceTests()
    {
        var options = Options.Create(new OrderingOptions { MaxLinesPerOrder = 10, EnableFraudScreening = true });
        _sut = new OrderService(_repository.Object, _fraud.Object, options, NullLogger<OrderService>.Instance);
    }

    [Fact]
    public async Task SubmitAsync_WhenOrderMissing_ReturnsNotFound()
    {
        _repository
            .Setup(r => r.FindAsync(It.IsAny<Guid>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync((Order?)null);

        var result = await _sut.SubmitAsync(new SubmitOrderCommand(Guid.NewGuid(), "key-1"), CancellationToken.None);

        result.IsFailure.Should().BeTrue();
        result.Error.Code.Should().Be("order.not_found");
        _repository.VerifyAll();
    }

    [Theory]
    [InlineData(1, true)]
    [InlineData(9, true)]
    [InlineData(11, false)]
    public async Task SubmitAsync_EnforcesLineLimit(int lineCount, bool expectedSuccess)
    {
        var order = new Order("cust-42");
        for (var i = 0; i < lineCount; i++)
        {
            order.AddLine($"AB-{1000 + i}", 1, new Money(10m, "EUR"));
        }

        _repository.Setup(r => r.FindAsync(order.Id, It.IsAny<CancellationToken>())).ReturnsAsync(order);
        _repository.Setup(r => r.SaveChangesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(1);
        _fraud.Setup(f => f.ScreenAsync(order, It.IsAny<CancellationToken>()))
              .Returns(ValueTask.FromResult(Result<bool>.Success(true)));

        var result = await _sut.SubmitAsync(new SubmitOrderCommand(order.Id, "key-2"), CancellationToken.None);

        result.IsSuccess.Should().Be(expectedSuccess);
    }

    [Fact]
    public void AddLine_OnSubmittedOrder_Throws()
    {
        var order = new Order("cust-7");
        order.AddLine("XY-2222", 2, new Money(5m, "EUR"));
        order.Submit();

        var act = () => order.AddLine("XY-3333", 1, new Money(5m, "EUR"));

        act.Should().Throw<InvalidOperationException>().WithMessage("*is Submitted, not Draft*");
    }

    [Fact]
    public void Money_AddAcrossCurrencies_Throws()
    {
        var euros = new Money(10m, "EUR");
        var dollars = new Money(10m, "USD");

        Assert.Throws<InvalidOperationException>(() => euros.Add(dollars));
    }
}

// ============================ Infrastructure/Json.cs ============================
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Contoso.Ordering.Infrastructure;

public sealed class MoneyJsonConverter : JsonConverter<Money>
{
    public override Money Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException($"Expected StartObject but found {reader.TokenType}.");
        }

        decimal amount = 0m;
        string currency = "EUR";

        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject)
            {
                return new Money(amount, currency);
            }

            if (reader.TokenType != JsonTokenType.PropertyName)
            {
                continue;
            }

            var propertyName = reader.GetString();
            reader.Read();

            switch (propertyName)
            {
                case "amount":
                    amount = reader.GetDecimal();
                    break;
                case "currency":
                    currency = reader.GetString() ?? "EUR";
                    break;
                default:
                    reader.Skip();
                    break;
            }
        }

        throw new JsonException("Unexpected end of JSON payload while reading Money.");
    }

    public override void Write(Utf8JsonWriter writer, Money value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WriteNumber("amount", value.Amount);
        writer.WriteString("currency", value.Currency);
        writer.WriteEndObject();
    }
}

[JsonSourceGenerationOptions(WriteIndented = false, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(OrderSummaryDto))]
[JsonSerializable(typeof(IReadOnlyList<OrderSummaryDto>))]
[JsonSerializable(typeof(ProblemDetails))]
internal sealed partial class OrderingJsonContext : JsonSerializerContext
{
}

// ============================ Infrastructure/Caching.cs ============================
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.Caching.Memory;

namespace Contoso.Ordering.Infrastructure;

public interface IPricingCache
{
    ValueTask<Money?> GetAsync(string sku, CancellationToken cancellationToken);
    ValueTask SetAsync(string sku, Money price, TimeSpan ttl, CancellationToken cancellationToken);
    void Invalidate(string sku);
}

public sealed class TwoTierPricingCache : IPricingCache, IDisposable
{
    private readonly IMemoryCache _local;
    private readonly IDistributedCache _remote;
    private readonly SemaphoreSlim _stampedeGuard = new(initialCount: 8, maxCount: 8);
    private readonly ILogger<TwoTierPricingCache> _logger;
    private long _hits;
    private long _misses;

    public TwoTierPricingCache(IMemoryCache local, IDistributedCache remote, ILogger<TwoTierPricingCache> logger)
    {
        _local = local;
        _remote = remote;
        _logger = logger;
    }

    public double HitRatio => Volatile.Read(ref _hits) + Volatile.Read(ref _misses) == 0
        ? 0d
        : (double)Volatile.Read(ref _hits) / (Volatile.Read(ref _hits) + Volatile.Read(ref _misses));

    public async ValueTask<Money?> GetAsync(string sku, CancellationToken cancellationToken)
    {
        var key = KeyFor(sku);

        if (_local.TryGetValue<Money>(key, out var cached))
        {
            Interlocked.Increment(ref _hits);
            return cached;
        }

        await _stampedeGuard.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_local.TryGetValue(key, out cached))
            {
                Interlocked.Increment(ref _hits);
                return cached;
            }

            var payload = await _remote.GetAsync(key, cancellationToken).ConfigureAwait(false);
            if (payload is null || payload.Length == 0)
            {
                Interlocked.Increment(ref _misses);
                return null;
            }

            var price = JsonSerializer.Deserialize<Money>(payload);
            _local.Set(key, price, new MemoryCacheEntryOptions
            {
                AbsoluteExpirationRelativeToNow = TimeSpan.FromMinutes(2),
                Size = 1,
                Priority = CacheItemPriority.High
            });
            Interlocked.Increment(ref _hits);
            return price;
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "Corrupt cache payload for {Sku}; evicting", sku);
            await _remote.RemoveAsync(key, cancellationToken).ConfigureAwait(false);
            return null;
        }
        finally
        {
            _stampedeGuard.Release();
        }
    }

    public async ValueTask SetAsync(string sku, Money price, TimeSpan ttl, CancellationToken cancellationToken)
    {
        var key = KeyFor(sku);
        var payload = JsonSerializer.SerializeToUtf8Bytes(price);

        await _remote.SetAsync(key, payload, new DistributedCacheEntryOptions
        {
            AbsoluteExpirationRelativeToNow = ttl
        }, cancellationToken).ConfigureAwait(false);

        _local.Set(key, price, ttl > TimeSpan.FromMinutes(2) ? TimeSpan.FromMinutes(2) : ttl);
    }

    public void Invalidate(string sku) => _local.Remove(KeyFor(sku));

    private static string KeyFor(string sku) => string.Concat("pricing:v3:", sku.ToUpperInvariant());

    public void Dispose() => _stampedeGuard.Dispose();
}

// ============================ Infrastructure/Resilience.cs ============================
using Polly;
using Polly.CircuitBreaker;
using Polly.Retry;

namespace Contoso.Ordering.Infrastructure;

public static class ResiliencePipelines
{
    public static ResiliencePipeline<HttpResponseMessage> BuildCatalogPipeline(ILogger logger)
    {
        return new ResiliencePipelineBuilder<HttpResponseMessage>()
            .AddRetry(new RetryStrategyOptions<HttpResponseMessage>
            {
                MaxRetryAttempts = 4,
                BackoffType = DelayBackoffType.Exponential,
                UseJitter = true,
                Delay = TimeSpan.FromMilliseconds(200),
                ShouldHandle = new PredicateBuilder<HttpResponseMessage>()
                    .Handle<HttpRequestException>()
                    .HandleResult(response => (int)response.StatusCode >= 500 || response.StatusCode == HttpStatusCode.TooManyRequests),
                OnRetry = args =>
                {
                    logger.LogWarning("Catalog retry {Attempt} after {Delay}", args.AttemptNumber, args.RetryDelay);
                    return default;
                }
            })
            .AddCircuitBreaker(new CircuitBreakerStrategyOptions<HttpResponseMessage>
            {
                FailureRatio = 0.5,
                SamplingDuration = TimeSpan.FromSeconds(30),
                MinimumThroughput = 20,
                BreakDuration = TimeSpan.FromSeconds(15)
            })
            .AddTimeout(TimeSpan.FromSeconds(5))
            .Build();
    }
}

public sealed class CatalogClient
{
    private readonly HttpClient _http;
    private readonly ResiliencePipeline<HttpResponseMessage> _pipeline;

    public CatalogClient(HttpClient http, ILogger<CatalogClient> logger)
    {
        _http = http;
        _pipeline = ResiliencePipelines.BuildCatalogPipeline(logger);
    }

    public async Task<Result<Money>> GetPriceAsync(string sku, CancellationToken cancellationToken)
    {
        var response = await _pipeline.ExecuteAsync(
            static async (state, ct) => await state.http.GetAsync($"/catalog/skus/{state.sku}/price", ct).ConfigureAwait(false),
            (http: _http, sku),
            cancellationToken).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            return Result<Money>.Failure(new Error("catalog.unavailable", $"Catalog returned {(int)response.StatusCode}."));
        }

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        var price = await JsonSerializer.DeserializeAsync<Money>(stream, cancellationToken: cancellationToken).ConfigureAwait(false);
        return Result<Money>.Success(price);
    }
}

// ============================ Application/Validation.cs ============================
using FluentValidation;

namespace Contoso.Ordering.Application;

public sealed record CreateOrderRequest(string CustomerId, IReadOnlyList<CreateOrderLine> Lines, string Currency);
public sealed record CreateOrderLine(string Sku, int Quantity, decimal UnitPrice);

public sealed class CreateOrderRequestValidator : AbstractValidator<CreateOrderRequest>
{
    private static readonly string[] SupportedCurrencies = { "EUR", "USD", "GBP", "CHF" };

    public CreateOrderRequestValidator()
    {
        RuleFor(request => request.CustomerId)
            .NotEmpty().WithMessage("Customer identifier is required.")
            .MaximumLength(64)
            .Matches("^[a-zA-Z0-9._-]+$").WithMessage("Customer identifier contains illegal characters.");

        RuleFor(request => request.Currency)
            .NotEmpty()
            .Must(currency => SupportedCurrencies.Contains(currency, StringComparer.OrdinalIgnoreCase))
            .WithMessage(request => $"Currency '{request.Currency}' is not supported.");

        RuleFor(request => request.Lines)
            .NotEmpty().WithMessage("At least one line is required.")
            .Must(lines => lines.Count <= 100).WithMessage("Too many lines.")
            .Must(lines => lines.Select(l => l.Sku).Distinct(StringComparer.OrdinalIgnoreCase).Count() == lines.Count)
            .WithMessage("Duplicate SKUs are not allowed; merge the quantities instead.");

        RuleForEach(request => request.Lines).ChildRules(line =>
        {
            line.RuleFor(l => l.Quantity).InclusiveBetween(1, 9_999);
            line.RuleFor(l => l.UnitPrice).GreaterThan(0m).PrecisionScale(18, 4, ignoreTrailingZeros: true);
            line.RuleFor(l => l.Sku).NotEmpty().MaximumLength(64);
        });
    }
}

public sealed class ValidationBehavior<TRequest, TResponse> : IPipelineBehavior<TRequest, TResponse>
    where TRequest : notnull
{
    private readonly IEnumerable<IValidator<TRequest>> _validators;

    public ValidationBehavior(IEnumerable<IValidator<TRequest>> validators) => _validators = validators;

    public async Task<TResponse> Handle(TRequest request, RequestHandlerDelegate<TResponse> next, CancellationToken cancellationToken)
    {
        if (!_validators.Any())
        {
            return await next().ConfigureAwait(false);
        }

        var context = new ValidationContext<TRequest>(request);
        var failures = (await Task.WhenAll(_validators.Select(v => v.ValidateAsync(context, cancellationToken))).ConfigureAwait(false))
            .SelectMany(result => result.Errors)
            .Where(failure => failure is not null)
            .ToList();

        if (failures.Count != 0)
        {
            throw new ValidationException(failures);
        }

        return await next().ConfigureAwait(false);
    }
}

// ============================ Infrastructure/Concurrency.cs ============================
using System.Collections.Concurrent;

namespace Contoso.Ordering.Infrastructure;

/// <summary>
/// Coalesces concurrent identical requests so only one hits the backing store.
/// </summary>
public sealed class RequestCoalescer<TKey, TValue> where TKey : notnull
{
    private readonly ConcurrentDictionary<TKey, Lazy<Task<TValue>>> _inFlight = new();

    public async Task<TValue> GetOrAddAsync(TKey key, Func<TKey, CancellationToken, Task<TValue>> factory, CancellationToken cancellationToken)
    {
        var lazy = _inFlight.GetOrAdd(key, k => new Lazy<Task<TValue>>(
            () => factory(k, cancellationToken),
            LazyThreadSafetyMode.ExecutionAndPublication));

        try
        {
            return await lazy.Value.ConfigureAwait(false);
        }
        finally
        {
            _inFlight.TryRemove(new KeyValuePair<TKey, Lazy<Task<TValue>>>(key, lazy));
        }
    }
}

public sealed class ShardedCounter
{
    private readonly PaddedLong[] _shards;

    [StructLayout(LayoutKind.Explicit, Size = 64)]
    private struct PaddedLong
    {
        [FieldOffset(0)] public long Value;
    }

    public ShardedCounter(int shardCount = 0)
    {
        var count = shardCount > 0 ? shardCount : Environment.ProcessorCount;
        _shards = new PaddedLong[count];
    }

    public void Increment()
    {
        var index = Environment.CurrentManagedThreadId % _shards.Length;
        Interlocked.Increment(ref _shards[index].Value);
    }

    public long Read()
    {
        long total = 0;
        for (var i = 0; i < _shards.Length; i++)
        {
            total += Interlocked.Read(ref _shards[i].Value);
        }
        return total;
    }
}

// ============================ Api/Hubs.cs ============================
using Microsoft.AspNetCore.SignalR;

namespace Contoso.Ordering.Api;

public interface IOrderClient
{
    Task OrderStatusChanged(Guid orderId, string status, DateTimeOffset changedUtc);
    Task ShipmentTracked(Guid orderId, string carrier, string trackingNumber);
}

[Authorize]
public sealed class OrderHub : Hub<IOrderClient>
{
    private readonly IOrderRepository _repository;
    private readonly ILogger<OrderHub> _logger;

    public OrderHub(IOrderRepository repository, ILogger<OrderHub> logger)
    {
        _repository = repository;
        _logger = logger;
    }

    public override async Task OnConnectedAsync()
    {
        var customerId = Context.User?.FindFirst("customer_id")?.Value;
        if (!string.IsNullOrEmpty(customerId))
        {
            await Groups.AddToGroupAsync(Context.ConnectionId, GroupFor(customerId)).ConfigureAwait(false);
            _logger.LogInformation("Connection {ConnectionId} joined {Group}", Context.ConnectionId, GroupFor(customerId));
        }
        await base.OnConnectedAsync().ConfigureAwait(false);
    }

    public override async Task OnDisconnectedAsync(Exception? exception)
    {
        if (exception is not null)
        {
            _logger.LogWarning(exception, "Connection {ConnectionId} dropped", Context.ConnectionId);
        }
        await base.OnDisconnectedAsync(exception).ConfigureAwait(false);
    }

    public async Task<IEnumerable<OrderSummaryDto>> Subscribe(string customerId, CancellationToken cancellationToken)
    {
        var orders = await _repository.ListByCustomerAsync(customerId, cancellationToken).ConfigureAwait(false);
        return orders.Select(o =>
        {
            var total = o.Total("EUR");
            return new OrderSummaryDto(o.Id, o.CustomerId, o.Status.ToString(), total.Amount, total.Currency, o.Lines.Count);
        });
    }

    private static string GroupFor(string customerId) => $"customer:{customerId}";
}

// ============================ Infrastructure/Migrations.cs ============================
using Microsoft.EntityFrameworkCore.Migrations;

namespace Contoso.Ordering.Infrastructure.Migrations;

public sealed partial class AddOutboxAndIndexes : Migration
{
    protected override void Up(MigrationBuilder migrationBuilder)
    {
        migrationBuilder.EnsureSchema(name: "ordering");

        migrationBuilder.CreateTable(
            name: "outbox",
            schema: "ordering",
            columns: table => new
            {
                sequence = table.Column<long>(type: "bigint", nullable: false)
                    .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                type = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: false),
                payload = table.Column<string>(type: "jsonb", nullable: false),
                enqueued_utc = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                dispatched_utc = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: true)
            },
            constraints: table =>
            {
                table.PrimaryKey("pk_outbox", x => x.sequence);
            });

        migrationBuilder.CreateIndex(
            name: "ix_outbox_pending",
            schema: "ordering",
            table: "outbox",
            column: "sequence",
            filter: "dispatched_utc IS NULL");

        migrationBuilder.Sql(@"
            CREATE OR REPLACE FUNCTION ordering.touch_updated_utc()
            RETURNS trigger AS $$
            BEGIN
                NEW.updated_utc = now();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;");
    }

    protected override void Down(MigrationBuilder migrationBuilder)
    {
        migrationBuilder.Sql("DROP FUNCTION IF EXISTS ordering.touch_updated_utc();");
        migrationBuilder.DropTable(name: "outbox", schema: "ordering");
    }
}

// ============================ Tests/IntegrationTests.cs ============================
using Microsoft.AspNetCore.Mvc.Testing;
using Testcontainers.PostgreSql;

namespace Contoso.Ordering.Tests;

public sealed class OrderingApiFixture : WebApplicationFactory<Program>, IAsyncLifetime
{
    private readonly PostgreSqlContainer _postgres = new PostgreSqlBuilder()
        .WithImage("postgres:16-alpine")
        .WithDatabase("ordering_tests")
        .WithUsername("test")
        .WithPassword("test")
        .WithCleanUp(true)
        .Build();

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Testing");
        builder.ConfigureServices(services =>
        {
            services.RemoveAll<DbContextOptions<OrderingDbContext>>();
            services.AddDbContext<OrderingDbContext>(options => options.UseNpgsql(_postgres.GetConnectionString()));
            services.RemoveAll<IFraudScreener>();
            services.AddSingleton<IFraudScreener, AlwaysAllowFraudScreener>();
        });
    }

    public async Task InitializeAsync()
    {
        await _postgres.StartAsync().ConfigureAwait(false);
        using var scope = Services.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<OrderingDbContext>();
        await db.Database.MigrateAsync().ConfigureAwait(false);
    }

    public new async Task DisposeAsync() => await _postgres.DisposeAsync().ConfigureAwait(false);
}

public sealed class OrdersEndpointTests : IClassFixture<OrderingApiFixture>
{
    private readonly HttpClient _client;

    public OrdersEndpointTests(OrderingApiFixture fixture) => _client = fixture.CreateClient();

    [Fact]
    public async Task GetById_UnknownOrder_Returns404WithProblemDetails()
    {
        var response = await _client.GetAsync($"/api/v1/orders/{Guid.NewGuid()}");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        var problem = await response.Content.ReadFromJsonAsync<ProblemDetails>();
        Assert.NotNull(problem);
        Assert.Equal(404, problem!.Status);
    }

    [Fact]
    public async Task Submit_WithoutIdempotencyKey_ReturnsValidationProblem()
    {
        var response = await _client.PostAsync($"/api/v1/orders/{Guid.NewGuid()}/submit", content: null);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }
}
'''


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def log_line(handle, text: str = "") -> None:
    print(text, flush=True)
    handle.write(text + "\n")
    handle.flush()


# --------------------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------------------

def build_corpus_shards() -> dict:
    """Deterministically derive the four hash-pinned corpus shards.

    Code: the embedded C# corpus split at the section boundary nearest the midpoint.
    Prose: two contiguous 120 kB slices of wiki.test.raw, each cut at a line boundary.
    """
    shards = {}

    body = CODE_CORPUS.replace("\r\n", "\n").strip("\n") + "\n"
    markers = [m.start() for m in re.finditer(r"^// =+ ", body, flags=re.M)]
    midpoint = len(body) // 2
    split_at = min(markers, key=lambda pos: abs(pos - midpoint)) if markers else midpoint
    shards["code_A"] = body[:split_at].encode("utf-8")
    shards["code_B"] = body[split_at:].encode("utf-8")

    if not os.path.isfile(PROSE_SRC):
        raise Refuse(
            f"prose corpus missing: {PROSE_SRC}\n"
            "  This is the WikiText-2 raw test split. Restore it (or point PROSE_SRC at a copy)\n"
            "  before running; the prereg pins its SHA-256 so a substitute will be rejected."
        )

    actual = sha256_file(PROSE_SRC)
    if actual != PROSE_SRC_SHA256:
        raise Refuse(
            f"prose corpus HASH MISMATCH for {PROSE_SRC}\n"
            f"  expected {PROSE_SRC_SHA256}\n"
            f"  actual   {actual}\n"
            "  The staked corpus changed under us; results would not be comparable to prereg #87."
        )

    with open(PROSE_SRC, "rb") as fh:
        raw = fh.read(PROSE_SHARD_BYTES * 2 + (1 << 16))

    def cut(buf: bytes, start: int, size: int) -> bytes:
        end = min(len(buf), start + size)
        nl = buf.rfind(b"\n", start, end)
        if nl > start:
            end = nl + 1
        return buf[start:end]

    a = cut(raw, 0, PROSE_SHARD_BYTES)
    b = cut(raw, len(a), PROSE_SHARD_BYTES)
    shards["prose_A"] = a
    shards["prose_B"] = b
    return shards


def emit_corpus(handle, force: bool = False) -> dict:
    os.makedirs(DATA, exist_ok=True)
    shards = build_corpus_shards()
    paths = {}
    for key, blob in shards.items():
        path = os.path.join(DATA, f"exp52_corpus_{key}.txt")
        digest = sha256_bytes(blob)
        if force or not os.path.isfile(path) or sha256_file(path) != digest:
            with open(path, "wb") as fh:
                fh.write(blob)
        paths[key] = path

        expected = CORPUS_SHA256.get(key)
        if expected is None:
            # KR-4 has no escape hatch. An unpinned shard used to log a warning and continue,
            # which meant deleting one line of CORPUS_SHA256 silently disabled the anti-fudge
            # gate for that shard. Refuse instead.
            raise Refuse(
                f"corpus shard {key} is NOT PINNED in CORPUS_SHA256 (sha256={digest}).\n"
                "  KR-4 requires every shard to be hash-pinned in the prereg before it is scored.\n"
                "  Add the hash to the prereg and to CORPUS_SHA256, or remove the shard."
            )
        if expected != digest:
            raise Refuse(
                f"corpus shard {key} HASH MISMATCH\n"
                f"  expected (staked in prereg #87) {expected}\n"
                f"  actual                          {digest}\n"
                "  The prompt set changed after staking. Restore it or re-stake a new prereg."
            )
        else:
            log_line(handle, f"  corpus {key:8s} {len(blob):8,d} B  sha256={digest[:16]}...  PINNED OK")
    return paths


# --------------------------------------------------------------------------------------
# imatrix driving + parsing
# --------------------------------------------------------------------------------------

def run_stamp(binary_sha, model_path, corpus_path, ctx, chunks, ngl, threads):
    """Everything that can change the counts. Two runs are comparable only if this matches.

    This is the C-14 analogue for a count measurement. Counts do not depend on clocks or
    thermals, so the *timing* form of C-14 does not bind -- but they DO depend on the binary,
    the tokenizer/model file, the corpus, the chunk budget, and (through CPU float reduction
    order at router near-ties) on the thread count and offload split. The idempotent cache made
    it easy to produce shard A under one setting and shard B under another and then compare
    them, which is exactly the cross-state comparison C-14 exists to forbid. The stamp is
    written beside every imatrix and re-checked on reuse.
    """
    st = os.stat(model_path)
    return {
        "binary_sha256": binary_sha,
        "model_path": model_path,
        "model_size": st.st_size,
        "model_mtime": int(st.st_mtime),
        "corpus_sha256": sha256_file(corpus_path),
        "ctx": int(ctx),
        "chunks": int(chunks),
        "ngl": int(ngl),
        "threads": int(threads) if threads else None,
    }


def run_imatrix(binary, model_path, corpus_path, out_path, ctx, chunks, ngl, threads, handle,
                force, binary_sha):
    stamp = run_stamp(binary_sha, model_path, corpus_path, ctx, chunks, ngl, threads)
    stamp_path = out_path + ".stamp.json"

    if os.path.isfile(out_path) and not force:
        # An imatrix produced under different settings is NOT a cache hit. Silently reusing one
        # was the sharpest defect in this script: `--chunks 2` once, defaults thereafter, and the
        # headline is scored on a 2-chunk file forever.
        if not os.path.isfile(stamp_path):
            raise Refuse(
                f"{os.path.basename(out_path)} exists but has no run stamp beside it, so the\n"
                "  settings that produced it are unknown and it cannot be trusted as a cache hit.\n"
                "  Re-run with --force to regenerate it."
            )
        with open(stamp_path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached != stamp:
            diff = [k for k in sorted(set(cached) | set(stamp)) if cached.get(k) != stamp.get(k)]
            raise Refuse(
                f"{os.path.basename(out_path)} was produced under DIFFERENT settings than this run.\n"
                f"  differing keys: {', '.join(diff)}\n"
                + "".join(f"    {k}: cached={cached.get(k)!r} now={stamp.get(k)!r}\n" for k in diff)
                + "  Mixing shards measured under different settings inside one comparison is the\n"
                  "  C-14 failure mode. Re-run with --force, or restore the original settings."
            )
        log_line(handle, f"  [skip] {os.path.basename(out_path)} already present, stamp matches "
                         "(idempotent; --force to redo)")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_out = out_path + ".partial"
    cmd = [
        binary,
        "-m", model_path,
        "-f", corpus_path,
        "-o", tmp_out,
        "--output-format", "gguf",
        "-c", str(ctx),
        "--chunks", str(chunks),
        "-ngl", str(ngl),
    ]
    if threads:
        cmd += ["-t", str(threads)]

    log_line(handle, f"  [run ] {' '.join(cmd)}")
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    elapsed = time.time() - started

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-25:])
        raise Refuse(
            f"llama-imatrix failed (exit {proc.returncode}) for {os.path.basename(out_path)}\n"
            f"--- last lines ---\n{tail}"
        )
    if not os.path.isfile(tmp_out):
        raise Refuse(f"llama-imatrix reported success but wrote no file: {tmp_out}")

    os.replace(tmp_out, out_path)
    with open(stamp_path, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, indent=1, sort_keys=True)
    log_line(handle, f"  [done] {os.path.basename(out_path)} in {elapsed:.1f}s")


def parse_counts(path, min_chunks, expect_ctx):
    """Return (per-layer expert count vectors, metadata) from an imatrix GGUF."""
    from gguf import GGUFReader
    import numpy as np

    reader = GGUFReader(path)

    def field(key):
        f = reader.fields.get(key)
        if f is None:
            return None
        try:
            return f.contents()
        except Exception:
            return None

    chunk_count = field("imatrix.chunk_count")
    chunk_size = field("imatrix.chunk_size")

    per_proj = {}
    for tensor in reader.tensors:
        m = COUNTS_RE.match(tensor.name)
        if not m:
            continue
        layer, proj = int(m.group(1)), m.group(2)
        per_proj.setdefault(proj, {})[layer] = np.asarray(tensor.data, dtype=np.float64).ravel().copy()

    if not per_proj:
        raise Refuse(
            f"no expert-count tensors in {os.path.basename(path)}\n"
            "  Expected blk.<N>.ffn_<gate|up|down>_exps.weight.counts. Either the model is not MoE\n"
            "  or this llama-imatrix build predates per-expert counts. Refusing to emit a number."
        )

    # G3a: the three projections see the SAME routing decisions. If they disagree the
    # instrument is broken and the run is VOID, not merely a failure.
    ref_proj = "down" if "down" in per_proj else sorted(per_proj)[0]
    layers = per_proj[ref_proj]
    for proj, table in per_proj.items():
        if proj == ref_proj:
            continue
        for layer, vec in table.items():
            if layer in layers and not np.array_equal(vec, layers[layer]):
                raise Refuse(
                    f"INSTRUMENT VOID in {os.path.basename(path)}: layer {layer} counts differ between "
                    f"ffn_{ref_proj}_exps and ffn_{proj}_exps. Routing counts must be identical across the "
                    "three expert projections; they are not, so the parse is not measuring what we think."
                )

    # G3c: every layer must carry nonzero routing mass. An all-zero count vector makes
    # share_of() return 0.0, which used to propagate to headline=0.0 and be reported as FAIL --
    # i.e. a broken parse would have been published as a REFUTATION. That must be VOID.
    for layer, vec in sorted(layers.items()):
        if not float(vec.sum()) > 0.0:
            raise Refuse(
                f"INSTRUMENT VOID in {os.path.basename(path)}: layer {layer} has zero total routing "
                "mass. A zero-mass layer scores 0.0 share, which would masquerade as a refutation. "
                "Refusing to emit a number."
            )

    # KR-3 requires >= min_chunks per shard. The old guard was `if chunk_count is not None and
    # ...`, so a build that does not write imatrix.chunk_count silently SKIPPED the kill rule and
    # a 1-chunk run would have scored. A missing field is now fatal.
    if chunk_count is None:
        raise Refuse(
            f"{os.path.basename(path)} has no imatrix.chunk_count field, so the KR-3 minimum-data "
            "check cannot be evaluated.\n  Refusing to score a run whose token budget is unknown "
            "(use a llama-imatrix build that writes it)."
        )
    if int(chunk_count) < min_chunks:
        raise Refuse(
            f"insufficient data in {os.path.basename(path)}: imatrix.chunk_count={int(chunk_count)} < "
            f"min {min_chunks}.\n  The corpus shard was shorter than the requested budget. Lower --chunks, "
            "lower --min-chunks deliberately, or lengthen the corpus."
        )
    if chunk_size is None:
        raise Refuse(
            f"{os.path.basename(path)} has no imatrix.chunk_size field, so the per-shard token "
            "budget cannot be confirmed. Refusing to score it."
        )
    if int(chunk_size) != int(expect_ctx):
        raise Refuse(
            f"{os.path.basename(path)} was measured at chunk_size={int(chunk_size)} but this run "
            f"requested ctx={int(expect_ctx)}.\n  The token budget differs from the one being "
            "compared against; that is a cross-state comparison. Re-run with --force."
        )

    meta = {
        "chunk_count": int(chunk_count) if chunk_count is not None else None,
        "chunk_size": int(chunk_size) if chunk_size is not None else None,
        "n_moe_layers": len(layers),
        "n_expert": int(next(iter(layers.values())).size) if layers else 0,
        "projections": sorted(per_proj),
    }
    return layers, meta


# --------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------

def hot_set(vec, k):
    """Indices of the k largest counts. Ties broken by expert index for determinism."""
    import numpy as np
    order = np.lexsort((np.arange(vec.size), -vec))
    return order[:k]


def share_of(vec, idx):
    total = float(vec.sum())
    if not total > 0:
        # Never silently return 0.0 -- that is indistinguishable from a genuine flat-skew
        # refutation. parse_counts already rejects zero-mass layers; this is the backstop.
        raise Refuse("zero routing mass in a scored layer; refusing to report 0.0 as a share.")
    return float(vec[idx].sum() / total)


def aggregate_share(rank_layers, score_layers, k):
    """Mass-weighted share across layers: rank on `rank_layers`, score on `score_layers`."""
    # Silently intersecting the two layer sets would score a subset of the model and report it
    # as the whole. If the ranking and scoring runs disagree on which layers exist, the parse is
    # wrong -- refuse rather than quietly drop layers.
    if set(rank_layers) != set(score_layers):
        missing = sorted(set(score_layers) - set(rank_layers))
        extra = sorted(set(rank_layers) - set(score_layers))
        raise Refuse(
            "INSTRUMENT VOID: ranking and scoring runs cover different MoE layer sets "
            f"(only-in-score={missing}, only-in-rank={extra}). Refusing to score the intersection "
            "and report it as the model."
        )

    selected = 0.0
    total = 0.0
    per_layer = {}
    for layer in sorted(score_layers):
        idx = hot_set(rank_layers[layer], k)
        vec = score_layers[layer]
        selected += float(vec[idx].sum())
        total += float(vec.sum())
        per_layer[layer] = share_of(vec, idx)
    if not total > 0:
        raise Refuse("zero total routing mass across all layers; refusing to emit a share.")
    return selected / total, per_layer


def analyse_model(runs, frac, handle):
    """runs: {(domain, shard): (layers, meta)} -> metrics dict."""
    import numpy as np

    any_meta = next(iter(runs.values()))[1]
    n_expert = any_meta["n_expert"]
    k = int(round(frac * n_expert))
    domains = sorted({d for (d, _) in runs})

    # The four runs of an arm must describe the SAME model. `next(iter(...))` used to take one
    # run's geometry and apply it to all; a stale cached file from a different model would have
    # been scored against the wrong expert count without a word.
    for (domain, shard), (_, meta) in sorted(runs.items()):
        if meta["n_expert"] != n_expert or meta["n_moe_layers"] != any_meta["n_moe_layers"]:
            raise Refuse(
                f"INSTRUMENT VOID: run {domain}_{shard} reports {meta['n_moe_layers']} layers x "
                f"{meta['n_expert']} experts but a sibling run reports "
                f"{any_meta['n_moe_layers']} x {n_expert}. These are not the same model."
            )

    # KR-2 needs at least two domains to have anything to cross. With one domain the cross-domain
    # dict is empty and retention used to default to 1.0 -- a vacuous PASS of the stability gate.
    if len(domains) < 2:
        raise Refuse(
            f"only {len(domains)} domain(s) present ({', '.join(domains) or 'none'}). KR-2 "
            "(cross-domain retention) cannot be evaluated, and an unevaluable kill rule must not "
            "be scored as passed. Restore the full prompt set."
        )
    for d in domains:
        for s in ("A", "B"):
            if (d, s) not in runs:
                raise Refuse(f"domain {d} is missing shard {s}; the held-out design is incomplete.")

    if not 0 < k < n_expert:
        raise Refuse(
            f"pinned set size k={k} out of range for n_expert={n_expert} (frac={frac}). "
            "A hot set of 0 or of every expert makes the metric meaningless."
        )

    # G3b: routing mass is n_tokens * expert_used_count in EVERY MoE layer, so it must be
    # identical across layers. This is what makes the index-chosen baseline exactly `frac`.
    # Compared with a relative tolerance rather than exact float equality: counts are integers
    # held as f32, so they are exact well past any budget we run, but a bit-equality test would
    # be needlessly brittle if that ever stopped holding.
    mass_uniform = True
    mass_report = {}
    for (domain, shard), (layers, _) in runs.items():
        totals = sorted({float(vec.sum()) for vec in layers.values()})
        mass_report[f"{domain}_{shard}"] = totals[:4]
        if totals and (totals[-1] - totals[0]) > 1e-6 * max(1.0, totals[-1]):
            mass_uniform = False

    metrics = {
        "n_expert": n_expert,
        "k_experts_pinned": k,
        "pinned_fraction_actual": k / n_expert,
        "n_moe_layers": any_meta["n_moe_layers"],
        "chunk_count": {f"{d}_{s}": m["chunk_count"] for (d, s), (_, m) in runs.items()},
        "routing_mass_uniform_across_layers": mass_uniform,
        "routing_mass_distinct_values": mass_report,
    }

    # in-sample (winner's curse; reported only as an inflated bound)
    metrics["in_sample_share"] = {
        d: aggregate_share(runs[(d, "A")][0], runs[(d, "A")][0], k)[0] for d in domains
    }

    # held-out, same domain: rank on A, score on B
    held_out = {}
    per_layer_curves = {}
    for d in domains:
        share, per_layer = aggregate_share(runs[(d, "A")][0], runs[(d, "B")][0], k)
        held_out[d] = share
        per_layer_curves[d] = per_layer
    metrics["held_out_same_domain_share"] = held_out
    metrics["per_layer_share"] = {d: {str(l): v for l, v in c.items()} for d, c in per_layer_curves.items()}

    # cross-domain: rank on A of one domain, score on B of the other
    cross = {}
    retention = {}
    for a in domains:
        for b in domains:
            if a == b:
                continue
            share, _ = aggregate_share(runs[(a, "A")][0], runs[(b, "B")][0], k)
            cross[f"{a}->{b}"] = share
            denom = held_out[b]
            retention[f"{a}->{b}"] = (share / denom) if denom > 0 else 0.0
    metrics["cross_domain_share"] = cross
    metrics["cross_domain_retention"] = retention

    # the deployable static pin: rank ONCE on the pooled A shards of every domain, then score
    # on each domain's held-out B shard. Fully out of sample.
    pooled_rank = {}
    for layer in runs[(domains[0], "A")][0]:
        acc = None
        for d in domains:
            vec = runs[(d, "A")][0].get(layer)
            if vec is None:
                continue
            acc = vec.copy() if acc is None else acc + vec
        if acc is not None:
            pooled_rank[layer] = acc

    pooled = {}
    pooled_per_layer = {}
    for d in domains:
        share, per_layer = aggregate_share(pooled_rank, runs[(d, "B")][0], k)
        pooled[d] = share
        pooled_per_layer[d] = per_layer
    metrics["pooled_static_pin_share"] = pooled
    metrics["pooled_static_pin_per_layer"] = {d: {str(l): v for l, v in c.items()} for d, c in pooled_per_layer.items()}

    headline = min(pooled.values())
    metrics["headline_worst_domain_share"] = headline
    metrics["headline_worst_domain"] = min(pooled, key=pooled.get)
    # The structural null is the fraction we ACTUALLY pin (k/n_expert = 41/128 = 0.3203), not the
    # nominal 0.32, and it must track --frac rather than the module constant -- otherwise a
    # --frac sweep would divide by a 0.32 baseline it never spent.
    metrics["baseline_index_share"] = k / n_expert
    metrics["uplift_vs_index_baseline"] = headline / (k / n_expert)

    # depth gradient
    depth = {}
    for d, curve in pooled_per_layer.items():
        ordered = [curve[l] for l in sorted(curve)]
        if ordered:
            half = len(ordered) // 2
            early = sum(ordered[:half]) / max(1, half)
            late = sum(ordered[half:]) / max(1, len(ordered) - half)
            depth[d] = {"early_half": early, "late_half": late, "late_minus_early": late - early}
    metrics["depth_gradient"] = depth

    # dispersion of the headline across layers -> is the verdict comfortable or marginal?
    worst = metrics["headline_worst_domain"]
    vals = [pooled_per_layer[worst][l] for l in sorted(pooled_per_layer[worst])]
    if len(vals) > 1:
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        metrics["headline_layer_sem"] = (var ** 0.5) / (len(vals) ** 0.5)
    else:
        metrics["headline_layer_sem"] = 0.0

    # never-routed experts (free eviction candidates, reported for task #55)
    unused = {}
    for d in domains:
        layers = runs[(d, "B")][0]
        zeros = sum(int((vec == 0).sum()) for vec in layers.values())
        unused[d] = zeros / max(1, len(layers) * n_expert)
    metrics["never_routed_fraction"] = unused

    return metrics


def verdict_for(metrics):
    """Score the staked kill rule. Returns (status, list of reason strings)."""
    reasons = []

    if not metrics["routing_mass_uniform_across_layers"]:
        reasons.append(
            "G3b VOID: routing mass is not identical across MoE layers, so the index-chosen "
            "baseline is not exactly 32% and the comparison is not the staked one."
        )
        return "VOID", reasons

    headline = metrics["headline_worst_domain_share"]
    if not metrics["cross_domain_retention"]:
        # Never default an unevaluable kill rule to "passed". analyse_model already refuses on a
        # single domain; this is the backstop.
        reasons.append(
            "G2 VOID: no cross-domain pair was computed, so the stability kill rule could not be "
            "evaluated. An unevaluable gate is not a passed gate."
        )
        return "VOID", reasons
    retention = min(metrics["cross_domain_retention"].values())

    g1 = headline >= GATE_G1_MIN_SHARE
    g2 = retention >= GATE_G2_MIN_RETENTION

    reasons.append(
        f"G1 held-out pooled hot-set share on worst domain ({metrics['headline_worst_domain']}) = "
        f"{headline:.4f} vs gate >= {GATE_G1_MIN_SHARE:.2f} -> {'PASS' if g1 else 'FAIL'}"
    )
    reasons.append(
        f"G2 worst cross-domain retention = {retention:.4f} vs gate >= {GATE_G2_MIN_RETENTION:.2f} -> "
        f"{'PASS' if g2 else 'FAIL'}"
    )

    # MARGINAL is a statement about G1's precision, and it may only be reached when the verdict
    # actually HINGES on G1. Previously this returned MARGINAL whenever the headline sat in the
    # 0.53-0.57 band regardless of G2, which meant KR-2 could not fire anywhere in that band --
    # a retention of 0.20 alongside a headline of 0.556 would have been reported as "too close to
    # call" rather than as the decisive stability refutation it is.
    if not g2:
        reasons.append(
            f"G2 fails decisively (retention {retention:.4f} < {GATE_G2_MIN_RETENTION:.2f}); the "
            "hot set is a fingerprint of the calibration domain. This settles the verdict "
            "regardless of where G1 landed, so the G1 marginality band does not apply."
        )
        return "FAIL", reasons

    if abs(headline - GATE_G1_MIN_SHARE) < MARGIN_BAND:
        reasons.append(
            f"G1 is within +/-{MARGIN_BAND} of the gate -> MARGINAL. Rerun with more chunks before "
            "treating either direction as settled."
        )
        return "MARGINAL", reasons

    return ("PASS" if g1 else "FAIL"), reasons


# --------------------------------------------------------------------------------------
# controls -- run with --selftest, no model and no GPU needed
# --------------------------------------------------------------------------------------
#
# The prereg claims a uniform-routing negative control and a domain-disjoint control were run
# before staking. Those claims had NO CODE in this script, so they were unreproducible and
# unfalsifiable. They are implemented here and drive the SAME analyse_model()/verdict_for() path
# the real arms drive. A positive control is included too: without one, "the gate can fail" and
# "the gate always fails" are indistinguishable.

def _synthetic_runs(kind, n_expert=128, n_layers=8, top_k=8, n_tokens=4096, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    domains = ["code", "prose"]
    k = int(round(FRAC * n_expert))
    mass = n_tokens * top_k

    # `partial_overlap` is the control that matters most: a hot set that is mostly shared but
    # partly domain-specific. It is engineered so G1 PASSES comfortably while G2 fails -- the
    # exact configuration in which a strong headline would carry the decision on its own if
    # KR-2 did not exist. If this control ever comes back PASS, KR-2 has stopped doing work.
    shared = 28
    spec = k - shared
    hot, hot_mass = {
        "uniform": ({d: None for d in domains}, None),
        "shared_hot": ({d: np.arange(k) for d in domains}, 0.80),
        "disjoint_hot": ({"code": np.arange(k), "prose": np.arange(k, 2 * k)}, 0.80),
        "partial_overlap": ({
            "code": np.concatenate([np.arange(shared), np.arange(k, k + spec)]),
            "prose": np.concatenate([np.arange(shared), np.arange(k + spec, k + 2 * spec)]),
        }, 0.90),
    }[kind]

    runs = {}
    for d in domains:
        for shard in ("A", "B"):
            layers = {}
            for layer in range(n_layers):
                p = np.full(n_expert, 1.0 / n_expert)
                if hot[d] is not None:
                    idx = hot[d]
                    p[:] = (1.0 - hot_mass) / (n_expert - idx.size)
                    p[idx] = hot_mass / idx.size
                layers[layer] = rng.multinomial(mass, p).astype(np.float64)
            meta = {"chunk_count": n_tokens // 512, "chunk_size": 512,
                    "n_moe_layers": n_layers, "n_expert": n_expert,
                    "projections": ["down", "gate", "up"]}
            runs[(d, shard)] = (layers, meta)
    return runs


def selftest(handle):
    """Drive the real analysis on synthetic routing with known ground truth."""
    expectations = {
        # kind            expected verdict   what it proves
        "uniform": ("FAIL", "flat routing must score at the structural null and FAIL G1 -- "
                            "the metric is not inflated by construction"),
        "disjoint_hot": ("FAIL", "fully domain-disjoint hot sets must not pass"),
        "partial_overlap": ("FAIL", "THE KR-2 ISOLATION CONTROL: G1 must PASS and G2 must FAIL "
                                    "here. This is the only proof that the stability gate can "
                                    "overturn a strong headline on its own"),
        "shared_hot": ("PASS", "strong and domain-stable concentration must PASS -- the gate is "
                               "not unconditionally hostile and a positive result is reachable"),
    }
    # Sub-gate assertions. `partial_overlap` is only a valid isolation control if G1 genuinely
    # passes there; if it drifted to a G1 failure it would stop testing G2 at all.
    subgates = {"partial_overlap": (True, False)}

    log_line(handle, "## controls (synthetic routing, same analysis path as the real arms)")
    ok = True
    for kind, (want, why) in expectations.items():
        runs = _synthetic_runs(kind)
        metrics = analyse_model(runs, FRAC, handle)
        status, reasons = verdict_for(metrics)
        got_headline = metrics["headline_worst_domain_share"]
        got_ret = min(metrics["cross_domain_retention"].values())
        good = status == want
        if kind in subgates:
            want_g1, want_g2 = subgates[kind]
            good = good and (got_headline >= GATE_G1_MIN_SHARE) == want_g1 \
                        and (got_ret >= GATE_G2_MIN_RETENTION) == want_g2
        agree = "OK  " if good else "MISMATCH"
        ok = ok and good
        log_line(handle, f"  [{agree}] {kind:15s} headline={got_headline:.4f} "
                         f"retention={got_ret:.4f} uplift={metrics['uplift_vs_index_baseline']:.2f}x "
                         f"-> {status} (expected {want})")
        log_line(handle, f"            {why}")
        for reason in reasons:
            log_line(handle, f"            {reason}")
    log_line(handle, f"  CONTROLS: {'PASS' if ok else 'FAILED - do not trust any headline'}")
    return ok


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def preflight(args, handle):
    if sys.version_info < (3, 9):
        raise Refuse(f"Python 3.9+ required, running {sys.version.split()[0]}")

    try:
        import numpy  # noqa: F401
    except ImportError:
        raise Refuse("numpy is not importable. Install with: python -m pip install numpy")

    try:
        import gguf  # noqa: F401
    except ImportError:
        raise Refuse(
            "the 'gguf' python package is not importable, so the imatrix output cannot be read.\n"
            "  Install with: python -m pip install gguf"
        )

    binary = args.imatrix_bin
    if not os.path.isfile(binary):
        # The prereg names ONE binary (tools/llamacpp-b10098). Falling through to whatever
        # llama-imatrix happens to be on PATH silently swaps the instrument -- a different build
        # can tokenise differently or count differently, and the swap would be invisible in the
        # headline. Refuse; make the operator name the substitute explicitly.
        found = shutil.which("llama-imatrix") or shutil.which("llama-imatrix.exe")
        hint = f"\n  A different build IS on PATH at {found} -- pass it explicitly with\n" \
               f"  --imatrix-bin \"{found}\" if you intend to substitute the instrument." if found else ""
        raise Refuse(
            f"llama-imatrix not found at the staked path {args.imatrix_bin}.{hint}\n"
            "  Refusing to fall back to an unnamed binary on PATH: the instrument is part of the "
            "measurement."
        )
    binary_sha = sha256_file(binary)

    selected = {}
    for name, path in MODELS.items():
        if args.model and name not in args.model:
            continue
        if not os.path.isfile(path):
            raise Refuse(
                f"model GGUF missing: {path}\n"
                f"  Required for arm '{name}'. Restore it, or restrict the run with --model <name>."
            )
        selected[name] = path

    if not selected:
        raise Refuse(f"no models selected. Known arms: {', '.join(MODELS)}")

    log_line(handle, f"  binary   {binary}")
    log_line(handle, f"           sha256={binary_sha}")
    for name, path in selected.items():
        log_line(handle, f"  model    {name:22s} {path}  ({os.path.getsize(path):,d} B)")
    if args.model:
        log_line(handle, "  NOTE --model restricts the arms; the OVERALL verdict below covers only "
                         "the arms actually run.")
    return binary, binary_sha, selected


def budget_deviations(args):
    """Any departure from the staked measurement budget. Non-empty -> the run is a SWEEP."""
    dev = []
    if abs(args.frac - FRAC) > 1e-12:
        dev.append(f"frac={args.frac} (staked {FRAC})")
    if args.ctx != STAKED_CTX:
        dev.append(f"ctx={args.ctx} (staked {STAKED_CTX})")
    if args.chunks != STAKED_CHUNKS:
        dev.append(f"chunks={args.chunks} (staked {STAKED_CHUNKS})")
    if args.min_chunks != STAKED_MIN_CHUNKS:
        dev.append(f"min_chunks={args.min_chunks} (staked {STAKED_MIN_CHUNKS})")
    if args.ngl != STAKED_NGL:
        dev.append(f"ngl={args.ngl} (staked {STAKED_NGL})")
    return dev


def main():
    parser = argparse.ArgumentParser(description="prereg #87 / task #52: expert-usage skew")
    parser.add_argument("--imatrix-bin", default=DEFAULT_IMATRIX_BIN)
    parser.add_argument("--model", action="append", help="restrict to named arm(s); repeatable")
    parser.add_argument("--ctx", type=int, default=512, help="imatrix chunk size (default 512)")
    # 8 chunks x 512 tok = 4096 tokens per shard. Capped by the shortest shard: the code
    # corpus tokenises to ~5.0-5.4k tokens per shard (9-10 chunks), prose to ~28k. Equal
    # budget across domains is deliberate -- the cross-domain comparison must not be
    # confounded by one domain contributing more routing events than the other.
    parser.add_argument("--chunks", type=int, default=8, help="chunks per shard (default 8 = 4096 tok)")
    parser.add_argument("--min-chunks", type=int, default=8, help="refuse below this many chunks")
    parser.add_argument("--ngl", type=int, default=0, help="GPU layers; 0 = pure CPU (default)")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--frac", type=float, default=FRAC, help="residency budget (default 0.32)")
    parser.add_argument("--force", action="store_true", help="redo imatrix runs that already exist")
    parser.add_argument("--emit-corpus-only", action="store_true", help="write + hash the corpus, then stop")
    parser.add_argument("--selftest", action="store_true",
                        help="run the synthetic controls through the real analysis path, then stop")
    args = parser.parse_args()

    os.makedirs(DATA, exist_ok=True)
    log_path = os.path.join(DATA, "exp52_expert_usage_skew.log")
    json_path = os.path.join(DATA, "exp52_expert_usage_skew.json")

    with open(log_path, "w", encoding="utf-8") as handle:
        log_line(handle, f"# exp52 expert-usage skew | pre-registration #{PREREG_ID}")
        log_line(handle, f"# staked in {PREREG}")
        log_line(handle, f"# gates: G1 held-out worst-domain share >= {GATE_G1_MIN_SHARE:.2f} at "
                         f"frac={args.frac:.2f}; G2 cross-domain retention >= {GATE_G2_MIN_RETENTION:.2f}")
        log_line(handle, f"# started {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_line(handle)

        try:
            if args.selftest:
                return 0 if selftest(handle) else 1

            deviations = budget_deviations(args)
            if deviations:
                # KR-4: any departure from the staked budget is a SWEEP, and a sweep must never be
                # readable as the headline. It is labelled in the log, in every verdict, and in the
                # JSON, and it is written to a DIFFERENT file so it cannot overwrite the staked run.
                log_line(handle, "!! SWEEP MODE -- this run departs from the staked budget:")
                for d in deviations:
                    log_line(handle, f"     {d}")
                log_line(handle, "!! Its numbers are NOT the prereg #87 headline and must be reported")
                log_line(handle, "!! separately and labelled as a sweep (KR-4).")
                log_line(handle)
                json_path = os.path.join(DATA, "exp52_expert_usage_skew.SWEEP.json")

            log_line(handle, "## corpus")
            corpus = emit_corpus(handle, force=args.force)
            if args.emit_corpus_only:
                log_line(handle, "\n--emit-corpus-only: stopping before any model run.")
                return 0

            log_line(handle, "\n## preflight")
            binary, binary_sha, models = preflight(args, handle)

            # The controls run in the SAME session as the headline, every time. If the analysis
            # path cannot distinguish flat from concentrated routing, or cannot fire G2 on
            # domain-disjoint hot sets, no measured number from this run means anything.
            log_line(handle)
            if not selftest(handle):
                raise Refuse(
                    "the synthetic controls did not reproduce their known ground truth, so the "
                    "analysis path is not trustworthy. No measured verdict will be emitted."
                )

            results = {}
            for name, model_path in models.items():
                log_line(handle, f"\n## arm {name}")
                runs = {}
                for key, corpus_path in sorted(corpus.items()):
                    domain, shard = key.rsplit("_", 1)
                    out = os.path.join(IMAT_DIR, f"{name}__{domain}__{shard}.imatrix.gguf")
                    run_imatrix(binary, model_path, corpus_path, out,
                                args.ctx, args.chunks, args.ngl, args.threads, handle, args.force,
                                binary_sha)
                    runs[(domain, shard)] = parse_counts(out, args.min_chunks, args.ctx)

                meta = next(iter(runs.values()))[1]
                log_line(handle, f"  parsed: {meta['n_moe_layers']} MoE layers x {meta['n_expert']} experts "
                                 f"| chunks={meta['chunk_count']} x {meta['chunk_size']} tok "
                                 f"| projections={','.join(meta['projections'])}")

                metrics = analyse_model(runs, args.frac, handle)
                status, reasons = verdict_for(metrics)
                metrics["status"] = status
                metrics["reasons"] = reasons
                results[name] = metrics

                log_line(handle)
                log_line(handle, f"  pinning {metrics['k_experts_pinned']}/{metrics['n_expert']} experts "
                                 f"({metrics['pinned_fraction_actual']:.4f} of each layer)")
                log_line(handle, f"  routing mass identical across layers: "
                                 f"{metrics['routing_mass_uniform_across_layers']}")
                for d, v in sorted(metrics["in_sample_share"].items()):
                    log_line(handle, f"    in-sample   {d:6s} {v:.4f}   (inflated: ranked and scored on one shard)")
                for d, v in sorted(metrics["held_out_same_domain_share"].items()):
                    log_line(handle, f"    held-out    {d:6s} {v:.4f}   (rank shard A, score shard B)")
                for pair, v in sorted(metrics["cross_domain_share"].items()):
                    log_line(handle, f"    cross       {pair:16s} {v:.4f}  retention "
                                     f"{metrics['cross_domain_retention'][pair]:.4f}")
                for d, v in sorted(metrics["pooled_static_pin_share"].items()):
                    log_line(handle, f"    STATIC PIN  {d:6s} {v:.4f}   (rank pooled A, score B - the deployable number)")
                for d, g in sorted(metrics["depth_gradient"].items()):
                    log_line(handle, f"    depth       {d:6s} early {g['early_half']:.4f} -> late "
                                     f"{g['late_half']:.4f}  (delta {g['late_minus_early']:+.4f})")
                for d, v in sorted(metrics["never_routed_fraction"].items()):
                    log_line(handle, f"    never-routed experts on {d:6s} {v:.4f}")
                log_line(handle, f"  headline (worst domain) {metrics['headline_worst_domain_share']:.4f} "
                                 f"vs index baseline {metrics['baseline_index_share']:.4f} "
                                 f"-> uplift {metrics['uplift_vs_index_baseline']:.2f}x "
                                 f"(layer SEM {metrics['headline_layer_sem']:.4f})")
                log_line(handle)
                for reason in reasons:
                    log_line(handle, f"  {reason}")
                log_line(handle, f"  ARM VERDICT: {status}")

            statuses = {n: m["status"] for n, m in results.items()}
            if "VOID" in statuses.values():
                overall = "VOID"
            elif "MARGINAL" in statuses.values():
                overall = "MARGINAL"
            elif all(s == "PASS" for s in statuses.values()):
                overall = "PASS"
            elif all(s == "FAIL" for s in statuses.values()):
                overall = "FAIL"
            else:
                overall = "SPLIT"

            if deviations:
                overall = f"SWEEP({overall})"

            payload = {
                "prereg": PREREG_ID,
                "prereg_path": PREREG,
                "is_staked_headline": not deviations and not args.model,
                "budget_deviations": deviations,
                "arms_restricted_to": args.model,
                "gates": {
                    "G1_min_share": GATE_G1_MIN_SHARE,
                    "G2_min_retention": GATE_G2_MIN_RETENTION,
                    "frac": args.frac,
                    "margin_band": MARGIN_BAND,
                },
                "config": {
                    "ctx": args.ctx, "chunks": args.chunks, "min_chunks": args.min_chunks,
                    "ngl": args.ngl, "threads": args.threads,
                    "imatrix_bin": binary, "imatrix_bin_sha256": binary_sha,
                },
                "corpus_sha256": {k: sha256_file(v) for k, v in corpus.items()},
                "arms": results,
                "overall": overall,
                "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1, sort_keys=True)

            log_line(handle)
            log_line(handle, "=" * 78)
            for name, status in sorted(statuses.items()):
                log_line(handle, f"  {name:22s} {status}")
            log_line(handle, f"  OVERALL: {overall}")
            log_line(handle, "=" * 78)
            if deviations:
                log_line(handle, "  SWEEP: not the prereg #87 headline. Report separately, labelled (KR-4).")
            elif overall == "PASS":
                log_line(handle, "  Hot-expert selection clears the staked gate. NOTE the implementability")
                log_line(handle, "  constraint in prereg #87: llama.cpp fuses all experts of a layer into one 3D")
                log_line(handle, "  tensor, so -ot CANNOT express this. A pass authorises a runtime change, not")
                log_line(handle, "  a regex change.")
            elif overall == "FAIL":
                log_line(handle, "  REFUTED for our models at this budget: keep the static -ot regex, and")
                log_line(handle, "  task #55 (cache-aware dropping) loses the gate it was waiting on.")
            elif overall == "SPLIT":
                # A split must never be summarised by its winning arm. KR-1 is per-model, so the
                # FAILing arm is a published refutation for that model in its own right.
                failing = sorted(n for n, s in statuses.items() if s == "FAIL")
                passing = sorted(n for n, s in statuses.items() if s == "PASS")
                log_line(handle, "  SPLIT: the kill rule is per-model and it FIRED on "
                                 f"{', '.join(failing)}.")
                log_line(handle, f"  Hot-expert caching is REFUTED for {', '.join(failing)} at this budget")
                log_line(handle, f"  and survives only on {', '.join(passing)}. Report BOTH at equal")
                log_line(handle, "  prominence; a split does not license a general claim, and task #55 stays")
                log_line(handle, "  gated to the surviving arm alone.")
            elif overall == "MARGINAL":
                log_line(handle, "  MARGINAL (KR-5): within +/-{:.2f} of the gate. NOT a verdict in either"
                                 .format(MARGIN_BAND))
                log_line(handle, "  direction. The staked remedy is to lengthen the C# corpus and re-stake.")
            elif overall == "VOID":
                log_line(handle, "  VOID (KR-3): the instrument did not validate. No verdict, no register")
                log_line(handle, "  entry. A void is not a failure and must not be reported as one.")
            log_line(handle, f"\n  raw -> {json_path}")
            log_line(handle, f"  log -> {log_path}")

            return 0 if overall in ("PASS", "FAIL") else 2

        except Refuse as exc:
            log_line(handle)
            log_line(handle, "REFUSING TO RUN -- a precondition is missing, so no number is produced.")
            log_line(handle, str(exc))
            return 3


if __name__ == "__main__":
    sys.exit(main())

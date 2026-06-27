# Angular Development Guidelines
Mandatory standards for modern Angular: standalone-by-default, signal-driven, zoneless-ready, strictly typed. Angular 20, TypeScript 5.8, signals, new control flow, inject(), typed forms.

---
name: angular
title: Angular Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [angular@20, typescript@5.8, "@angular/cli@20", vitest, jest, "@ngrx/signals@19", angular-eslint, prettier, npm@10]
requires:
  - typescript
  - tdd
  - secure-coding
recommends:
  - rest
  - accessibility
  - e2e-testing
  - ui
  - css
  - html
  - performance
  - zod
  - observability
provides:
  - angular-standalone
  - angular-signals
  - angular-control-flow
  - angular-di
  - typed-forms
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Angular. The language itself (types, generics, strict config) lives in [`typescript.md`](guides://typescript.md) and is not repeated here.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Angular code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`typescript.md`](guides://typescript.md) — the language: strict compiler flags, types, generics, no `any`. Angular adds `strictTemplates`; it is not a separate language.
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Angular binding: `TestBed` + component/service specs; runner is Jest or Vitest — Karma/Jasmine is deprecated since Angular 20; use CDK component harnesses (`@angular/cdk/testing`) for DOM assertions.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy, XSS/CSP. *(Angular binding: built-in contextual auto-sanitization; `DomSanitizer.bypassSecurityTrust*` is a last resort that MUST be justified; `npm audit`.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`rest.md`](guides://rest.md) — API design *(binding: `HttpClient`, `provideHttpClient(withFetch())`, functional interceptors, `httpResource`)*
> - [`accessibility.md`](guides://accessibility.md) — WCAG, ARIA, focus *(binding: Angular CDK a11y — `LiveAnnouncer`, `FocusTrap`, `cdkTrapFocus`)*
> - [`e2e-testing.md`](guides://e2e-testing.md) — Playwright/Cypress browser flows.
> - [`ui.md`](guides://ui.md) · [`css.md`](guides://css.md) · [`html.md`](guides://html.md) — component design & styling (binding: view-encapsulated SCSS, Angular Material/CDK).
> - [`performance.md`](guides://performance.md) — budgets, Core Web Vitals *(binding: lazy routes, OnPush/zoneless, `@defer`, `NgOptimizedImage`)*
> - [`zod.md`](guides://zod.md) — runtime validation of HTTP payloads at the boundary.
> - [`observability.md`](guides://observability.md) · [`error-handling.md`](guides://error-handling.md) — telemetry & error strategy *(binding: `ErrorHandler`, error interceptor)*

> 📎 **SEE ALSO:** [`comments.md`](guides://comments.md) · [`semver.md`](guides://semver.md) · [`ci-cd.md`](guides://ci-cd.md)

---

## 1. Core Philosophies: ANGULAR-FIRST

Angular-specific principles only. TDD, security, typing, and accessibility come from §0.

- **A**lways standalone: components/directives/pipes are standalone by default (Angular 19+); `standalone: true` is implicit — **never** write NgModules in new code. Bootstrap with `bootstrapApplication`.
- **N**ew control flow: `@if`/`@for`/`@switch`/`@defer` in templates — never `*ngIf`/`*ngFor`/`*ngSwitch` (legacy). `@for` MUST declare `track`.
- **G**ranular reactivity: signals (`signal`/`computed`/`effect`) are the default state primitive; signal `input()`/`output()`/`model()` over `@Input`/`@Output` decorators. Derive with `computed`, don't recompute in templates.
- **U**nidirectional injection: dependencies via `inject()`; functional guards, resolvers, and interceptors over class-based ones.
- **L**ean change detection: `OnPush` everywhere, targeting zoneless (`provideZonelessChangeDetection`). No manual `markForCheck` churn — let signals drive it.
- **A**OT & typed: production is AOT with `strictTemplates`; typed reactive forms; no `any` escapes (see `typescript.md`).
- **R**eactive interop: signals for state, RxJS for event streams; bridge with `toSignal`/`toObservable`; auto-unsubscribe with `takeUntilDestroyed`.

**Verified Code**: Agent-generated Angular MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `NG-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| NG-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `ng test --watch=false` | exit 0, 0 skips |
| NG-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `ng test --watch=false` | failing→passing |
| NG-TST-03 | Coverage MUST meet the project gate | `ng test --watch=false --code-coverage` | ≥ threshold |
| NG-FMT-01 | Code MUST be formatted | `npx prettier --check .` | no diff |
| NG-LINT-01 | Lint MUST pass (TS + template) | `ng lint` | exit 0 |
| NG-TYP-01 | Strict TS + `strictTemplates`, no `any` (see `typescript.md`) | `ng build` / `tsc --noEmit` | exit 0 |
| NG-STD-01 | No NgModules in new code; standalone only | `grep -r "@NgModule" src/` | none (legacy-only) |
| NG-CF-01 | Templates MUST use `@if`/`@for`/`@switch`, not `*ngIf`/`*ngFor` | `grep -rE "\*ng(If\|For\|Switch)" src/` | none |
| NG-CD-01 | Components MUST be `OnPush` (zoneless target) | review / lint rule | no default CD |
| NG-FORM-01 | Reactive forms MUST be typed (no untyped `FormGroup`) | review / `tsc` | typed controls |
| NG-A11Y-01 | UI MUST meet WCAG 2.2 AA (see `accessibility.md`) | axe / Lighthouse a11y | ≥ project gate |
| NG-PERF-01 | Feature routes lazy-loaded; budgets enforced (see `performance.md`) | `ng build` (budgets) | within budget |
| NG-SEC-01 | No unjustified `bypassSecurityTrust*`; sanitize dynamic HTML (see `secure-coding.md`) | `grep -r "bypassSecurityTrust" src/` | each justified |
| NG-SEC-02 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `npm audit --audit-level=high` | 0 high/critical |
| NG-DEP-01 | Lockfile committed & reproducible | `npm ci` | installs clean |
| NG-DOC-01 | Public components/services/APIs documented (see `comments.md`) | `npm run docs` / TypeDoc | builds clean |

> **Forbidden**: NgModules in new code, `*ngIf`/`*ngFor`, untyped forms, raw `any`, subscribing without teardown, `bypassSecurityTrust*` without a written reason, mutating signal values in place (always `set`/`update` with new references), shipping implementation before its test (violates `tdd.md`).

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
npx prettier --check .                       # NG-FMT-01
ng lint                                      # NG-LINT-01
ng build                                     # NG-TYP-01 (strictTemplates) + NG-PERF-01 budgets
ng test --watch=false --code-coverage        # NG-TST-01/02/03
npm audit --audit-level=high                 # NG-SEC-02
npm ci                                       # NG-DEP-01 (clean lockfile install)
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Feature-first layout; lazy boundaries map to route groups. Architectural *principles* (layering, dependency direction) are owned by `typescript.md`/architecture guides — below is only the Angular mapping.

```
src/
├── app/
│   ├── core/            # app-singletons: guards, interceptors, root services, models
│   ├── shared/          # reusable standalone components, directives, pipes, validators
│   ├── features/<feat>/ # lazy-loaded: <feat>.routes.ts + components + services + store
│   ├── layout/          # shell: header, footer, sidebar
│   ├── app.component.ts
│   ├── app.config.ts    # ApplicationConfig: providers (router, http, zoneless…)
│   └── app.routes.ts    # root routes (lazy loadComponent/loadChildren)
├── environments/
└── main.ts              # bootstrapApplication(AppComponent, appConfig)
```

- One concern per file; standalone classes export their own dependencies via `imports`.
- Path aliases (`@core/*`, `@shared/*`, `@features/*`) in `tsconfig.json` `paths`.
- No circular feature deps; `core` and `shared` never import from `features`.

---

## 5. Angular Specifics

The unique value of this guide. Code blocks illustrate Angular idioms, not generic concepts.

### A. Standalone components & signal I/O
`standalone: true` is the Angular 19+ default and may be omitted. Inputs/outputs are signals; default to `OnPush`.

```typescript
import { Component, ChangeDetectionStrategy, input, output, computed, model } from '@angular/core';

@Component({
  selector: 'app-user-card',
  imports: [RouterLink],                       // import deps directly — no NgModule
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <h3>{{ user().name }}</h3>
    <img [ngSrc]="avatar()" [alt]="user().name" width="48" height="48" />
    @if (editable()) { <button type="button" (click)="edit.emit(user().id)">Edit</button> }
  `,
})
export class UserCardComponent {
  user = input.required<User>();               // signal input (typed, required)
  editable = input(false);                      // optional with default
  selected = model(false);                      // two-way: [(selected)]
  edit = output<string>();                       // signal output (no EventEmitter)
  avatar = computed(() => this.user().avatar ?? '/assets/default-avatar.png');
}
```
Legacy `@Input()/@Output()/EventEmitter` still compile but MUST NOT be used in new code. Map route params straight to inputs with `provideRouter(routes, withComponentInputBinding())`.

### B. New control flow (templates)
Built-in, no imports. `@for` **requires** `track`; `@defer` lazy-loads a block on a trigger.

```html
@if (loading()) { <app-spinner /> }
@else if (error()) { <app-error [msg]="error()" /> }
@else {
  @for (u of users(); track u.id) { <app-user-card [user]="u" /> }
  @empty { <p>No users</p> }
}

@switch (status()) {
  @case ('ok') { <app-ok /> }
  @default { <app-unknown /> }
}

@defer (on viewport) { <app-heavy-chart [data]="data()" /> }
@placeholder { <div class="skeleton"></div> }
@loading (minimum 200ms) { <app-spinner /> }
```
`@defer` triggers: `on idle | viewport | interaction | hover | timer(…)` or `when <condition>`. Use it to code-split heavy, below-the-fold UI (see `performance.md`).

### C. Signals — state, derivation, effects
Signals are the default reactive primitive. Mutate with `set`/`update` and **new references** — never mutate arrays/objects in place.

```typescript
count = signal(0);
items = signal<Item[]>([]);
double = computed(() => this.count() * 2);                 // pure, cached, lazy
linkedFilter = linkedSignal(() => this.defaultFilter());   // resettable derived writable

add(i: Item) { this.items.update(xs => [...xs, i]); }      // new array reference

constructor() {
  effect(() => localStorage.setItem('count', `${this.count()}`)); // side effects only
}
```
- `computed` for derived state; never duplicate it in a field. `effect` is for side effects (DOM, storage, logging) — **not** for deriving state.
- `untracked(() => …)` to read a signal inside an effect without subscribing.
- `resource()`/`httpResource()` (Angular 19.2+) model async data as signals with `value`/`status`/`error` and built-in cancellation.

### D. Dependency injection — `inject()`
Field injection with `inject()`; reserve constructors for parameter properties only when needed.

```typescript
@Injectable({ providedIn: 'root' })
export class UserService {
  private http = inject(HttpClient);
  private cfg = inject(APP_CONFIG);
  private analytics = inject(AnalyticsService, { optional: true });
}

export interface AppConfig { apiUrl: string; }
export const APP_CONFIG = new InjectionToken<AppConfig>('app.config');   // typed token
// provide in app.config.ts: { provide: APP_CONFIG, useValue: environment }
```
`inject()` only runs in an injection context (constructor/field initializer, factory, functional guard/resolver/interceptor). `providedIn: 'root'` enables tree-shaking.

### E. Routing & lazy loading
Routes are data; lazy-load every feature with `loadComponent`/`loadChildren`. Guards/resolvers are functions.

```typescript
export const routes: Routes = [
  { path: 'home', loadComponent: () => import('./features/home/home.component').then(m => m.HomeComponent) },
  { path: 'admin', canActivate: [authGuard, hasRole(['admin'])],
    loadChildren: () => import('./features/admin/admin.routes').then(m => m.ADMIN_ROUTES) },
  { path: '**', loadComponent: () => import('./shared/not-found.component').then(m => m.NotFoundComponent) },
];

export const authGuard: CanActivateFn = (_route, state) => {
  const auth = inject(AuthService), router = inject(Router);
  return auth.isAuthenticated() || router.createUrlTree(['/login'], { queryParams: { returnUrl: state.url } });
};
export const userResolver: ResolveFn<User> = route => inject(UserService).getUser(route.paramMap.get('id')!);
```

### F. HttpClient, interceptors & RxJS interop
Configure `HttpClient` once with `provideHttpClient(withFetch(), withInterceptors([...]))`. Interceptors are functions.

```typescript
export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const token = inject(AuthService).getToken();
  return next(token ? req.clone({ setHeaders: { Authorization: `Bearer ${token}` } }) : req);
};

// Stream → signal; debounce/switchMap search; auto-teardown on destroy
results = signal<User[]>([]);
constructor() {
  this.searchControl.valueChanges.pipe(
    debounceTime(300), distinctUntilChanged(),
    switchMap(q => this.api.search(q ?? '')),
    takeUntilDestroyed(),                       // no manual unsubscribe
  ).subscribe(r => this.results.set(r));
}
// Or declaratively: data = toSignal(this.api.getUsers(), { initialValue: [] });
```
Validate untrusted payloads at the boundary with Zod (see `zod.md`) before trusting their types. API design/status-codes are owned by `rest.md`.

### G. Typed reactive forms
Forms MUST be typed; use `nonNullable` controls and `getRawValue()` for the typed snapshot.

```typescript
private fb = inject(NonNullableFormBuilder);
form = this.fb.group({
  email: this.fb.control('', [Validators.required, Validators.email]),
  password: this.fb.control('', [Validators.required, Validators.minLength(8)]),
  profile: this.fb.group({ firstName: this.fb.control(''), age: this.fb.control<number | null>(null) }),
});
// form.controls.email is FormControl<string>; form.getRawValue() is fully typed.
```
Custom validators are `ValidatorFn` factories returning `ValidationErrors | null`; cross-field validators read `control.parent`. Async validators return `Observable<ValidationErrors | null>`.

### H. Directives, pipes & content projection
Standalone directives/pipes; project content with `<ng-content>` and signal-based queries (`contentChild`, `viewChild`).

```typescript
@Directive({ selector: '[appHighlight]' })
export class HighlightDirective {
  color = input('yellow');
  private el = inject(ElementRef);
  @HostListener('mouseenter') on() { this.el.nativeElement.style.background = this.color(); }
}

@Pipe({ name: 'truncate' })                      // pure by default — cached, OnPush-friendly
export class TruncatePipe implements PipeTransform {
  transform(v: string, max = 50): string { return v.length > max ? v.slice(0, max) + '…' : v; }
}
```
Prefer `host` metadata / `HostBinding`/`HostListener` over touching the DOM directly; use `Renderer2` when you must, for SSR safety.

### I. Change detection — zoneless & OnPush
Target **zoneless**: `bootstrapApplication(App, { providers: [provideZonelessChangeDetection()] })` and drop `zone.js` from `polyfills`. Signals, `AsyncPipe`, and `markForCheck` schedule CD; everything stays `OnPush`. Avoid `ChangeDetectorRef.detectChanges()` loops and template method calls that do work (use `computed`).

### J. Lifecycle & cleanup
Use `inject(DestroyRef)` + `takeUntilDestroyed(destroyRef)` (or argument-less form in an injection context) to tear down subscriptions — never leak. Prefer `afterNextRender`/`afterRender` for DOM-measurement work over `ngAfterViewInit` (SSR-safe). Keep `ngOnInit` thin; do data setup via resolvers, `resource()`, or signal inputs.

---

## 6. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). Angular binding:

```bash
npm ci                        # reproducible install from package-lock.json (NG-DEP-01)
ng add @angular/material      # add Material/CDK (a11y-ready components)
ng update                     # framework migration + peer-dep sync (schematics auto-migrate)
npm audit --audit-level=high  # NG-SEC-02: CVE scan
```
Commit `package-lock.json`; use `npm ci` in CI. Run `ng update` for major bumps so schematics migrate code (e.g. control-flow, standalone). Keep `@angular/*`, `@angular/cli`, and `@angular/cdk` on the same major.

---

## 7. Quick Reference

```bash
ng serve                                   # dev server (http://localhost:4200)
ng generate component features/users/list  # scaffold (g c / g s / g d / g p / g g)
ng build --configuration production        # AOT prod build + budgets
ng test --watch=false --code-coverage      # test + coverage
ng lint && npx prettier --check .          # lint + format
ng update                                  # migrate to latest Angular
```

```typescript
user = input.required<User>();             // signal input
edit = output<string>();                    // signal output
total = computed(() => this.items().length);// derived state
data = toSignal(this.api.get(), { initialValue: [] });  // stream → signal
// template: @if / @for(track) / @switch / @defer
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] NG-FMT-01 — `prettier --check` clean
- [ ] NG-LINT-01 — `ng lint` clean (TS + template)
- [ ] NG-TYP-01 — strict TS + `strictTemplates`, no `any`
- [ ] NG-STD-01 — standalone only, no new NgModules
- [ ] NG-CF-01 — `@if`/`@for`/`@switch`, no `*ngIf`/`*ngFor`
- [ ] NG-CD-01 — `OnPush` (zoneless target)
- [ ] NG-FORM-01 — reactive forms typed
- [ ] NG-TST-01/02/03 — tests pass, bugs have regression tests, coverage ≥ gate
- [ ] NG-A11Y-01 — WCAG 2.2 AA verified
- [ ] NG-PERF-01 — routes lazy-loaded, budgets within limits
- [ ] NG-SEC-01 — no unjustified `bypassSecurityTrust*`, dynamic HTML sanitized
- [ ] NG-SEC-02 — `npm audit` 0 high/critical
- [ ] NG-DEP-01 — `package-lock.json` committed, `npm ci` clean
- [ ] NG-DOC-01 — public components/services documented
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Angular Guidelines**

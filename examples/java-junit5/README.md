# Java 示例：一个真的能跑起来的 Maven 仓库

这是 Java 工具链的**真机验收样本**，不是一份教学代码。它存在的理由只有一条：
Java 那条路径的罐装输出与整链测试可以一直是绿的，而镜像里的 `mvn -o` 到底跑不跑得起来，
只有在一台装了 Docker 的机器上跑一遍才知道。仓库里的 `make acceptance` 跑的就是它。

一个单模块 Maven 项目：JUnit 5 + JaCoCo，退款按明细拆分，二十来行。
插件版本与 `src/honest_ci/maven.py` 的 `MAVEN_PLUGINS` 对齐。

```
pom.xml                  依赖与插件（版本全部钉死，见下）
checkstyle.xml           Checkstyle 配置：小，只留几条干净代码上不会响的
pmd-ruleset.xml          PMD 规则集，同理
honest-ci-rules.toml     规则库：正则 / 路径 / 扫描器各一类，没有判断类规则
honest-ci-policy.toml    结构护栏：required_checks = ["review", "regression"]
scenarios/*.patch        三种终态各一份改动
src/main/java/...        RefundSplitter：整数分账
src/test/java/...        RefundSplitterTest：5 个用例
```

## 三种终态

| 改动 | 应得终态 | 退出码 | 由谁判的 |
|---|---|---|---|
| `scenarios/clean.patch`：加一个 `totalOf` 方法与它的测试 | `mergeable` | 0 | 评审门禁无发现；回归门禁 7 个用例全过、行覆盖率过线 |
| `scenarios/violation.patch`：新增的方法用 `double` 算金额 | `blocked` | 1 | 评审门禁的 R-01（high，确定性正则）命中新增行 |
| `scenarios/drop-test.patch`：删掉整个测试类 | `needs_human` | 3 | 回归门禁：测试跑了但收集到 0 个用例——没有回归证据，不是通过 |

第三行是这个示例最值得看的一行。删掉测试之后什么都不会「失败」：`mvn test` 退出 0，
评审门禁也没有可拦的新增行。把它读成「通过」需要的只是少问一句话。

## 自己跑一遍

```bash
cp -r examples/java-junit5 /tmp/honest-ci-java && cd /tmp/honest-ci-java && git init -q && git add -A && git commit -qm baseline
```

```bash
export HONEST_CI_MAVEN_EXTRA_ARTIFACTS=org.apache.maven.surefire:surefire-junit-platform:3.5.2,org.junit.platform:junit-platform-commons:1.9.3
```

```bash
honest-ci doctor --workspace /tmp/honest-ci-java
```

`doctor` 的「就绪检查」会说 Maven 本地仓库是空的。第一次 `verify` 会把它装起来——
这一步联网，而且是整条路径上**唯一**联网的一步。

```bash
git apply scenarios/clean.patch && honest-ci verify --workspace /tmp/honest-ci-java; echo "退出码 $?"
```

换另外两份补丁前先 `git checkout -- . && git clean -fd`。

## 三件在真机上才会发现的事

**一、冷缓存装配比扫描慢一个量级。** `go-offline` 在这个只有一个 JUnit 依赖的项目上要拉
130 MB 以上才轮到解析插件。扫描器的超时（`HONEST_CI_COMMAND_TIMEOUT_SECONDS`，默认 300 秒）
按「一次 ruff 跑 5 分钟就是出事了」来定，装配套不进去，所以装配有自己的
`HONEST_CI_PROVISION_TIMEOUT_SECONDS`（默认 1800 秒）。超时退出码是 124，错误里会直接这么说。

**二、有些构件 `go-offline` 预取不到。** Maven 有一类构件是目标执行时才决定的：
Surefire 按测试框架挑 provider，报告插件渲染时挑皮肤。两者都不在任何一份 pom 的依赖图里，
`go-offline` 和 `resolve-plugins` 都看不见（`resolve-plugins` 连 pom 里显式写给 surefire 的
`<dependencies>` 都不会拉）。离线跑到这里会停在一个谁也没声明过的下载上。三种收口：

- **皮肤是我们的问题**——我们钉死了 PMD 的版本，就得钉死它渲染时要的那张皮肤。
  已经在 `maven.py` 的 `MAVEN_RUNTIME_ARTIFACTS` 里，不用配。
- **JUnit Platform launcher 是项目的问题**——Surefire 没找到 launcher 时会自己补一个。
  本项目的 `pom.xml` 显式声明了它（JUnit 官方从 5.11 起也这么建议），版本跟着项目走，
  装配当普通测试依赖一起拉。
- **provider 和它自带的基线是这台机器的问题**——版本跟着项目选的 surefire 走，
  工具链猜不出来，所以写在 `HONEST_CI_MAVEN_EXTRA_ARTIFACTS` 里。注意是**两个**坐标：
  声明了 launcher 1.11.4 之后，Surefire 仍然去解析它自己那套基线的
  `org.junit.platform:junit-platform-commons:1.9.3`——pom 里写什么都不影响这一个。
  漏了也不会误判成「代码有问题」：离线解析失败会报成**工具不可用**
  （fail-closed 照样拦截），并把缺的坐标原样告诉你。

**三、上一次构建会冒充这一次的证据。** 把 `drop-test.patch` 打在一个刚跑过一次的工作区上：
测试源文件没了，但 `target/test-classes` 里那个 `.class` 还在——Maven 不会删一个源文件已经
消失的编译产物——于是 Surefire 照跑不误，套件为一棵树上没有的文件报出用例数。
所以每次运行在第一个 Maven 目标之前先 `clean` 一次（`maven-clean-plugin` 也钉了版本，
装配时一并拉下来）。代价是每次全量重编译；换来的是证据包里每个数字都出自被指纹化的那棵树。
这一条 `tests/integration/test_java_acceptance.py` 会显式检查：跑 `drop-test` 之前先断言
工作区里确实留着上一次的 `.class`，否则这条用例什么也测不到。

## 为什么没有判断类规则

判断类规则要模型凭据才能判定；没凭据时它记为「没判定」并转人工——那样这个示例永远到不了
`mergeable`，而到不了的原因跟 Java 工具链没有关系。验收要的是确定性。真实仓库的规则库
该有判断类规则，`honest-ci-rules.example.toml` 里有写法。

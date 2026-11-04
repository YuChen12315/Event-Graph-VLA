from flax import struct

from openpi.models.model import ArrayT
from openpi.models.model import Observation as _Observation
from openpi.models.model import preprocess_observation as _preprocess_observation
import openpi.shared.array_typing as at


#  b: batch size
#  t: time step
#  v: view
#  h: height
#  w: width
#  c: channel
#  l: seq len


@at.typecheck
@struct.dataclass
class HistAugObservation(_Observation):
    # for perceptual memory
    static_image_emb: at.Float[at.Array, "b l1 d1"] | None = None
    static_mask: at.Bool[at.Array, "b l1"] | None = None
    static_pos_emb: at.Float[at.Array, "b l1 d2"] | None = None
    static_state_emb: at.Float[at.Array, "b l1 d3"] | None = None

    # for graph+DPP learned selection (TCGM): candidate window frames (right-padded to cw)
    candidate_image_emb: at.Float[at.Array, "b cw cp cd1"] | None = None
    candidate_pos_emb: at.Float[at.Array, "b cw cp cd2"] | None = None
    candidate_state_emb: at.Float[at.Array, "b cw cd3"] | None = None
    candidate_mask: at.Bool[at.Array, "b cw"] | None = None

    # for recurrent memory
    recur_image_emb: at.Float[at.Array, "b t v p d1"] | None = None  # left padded
    recur_mask: at.Bool[at.Array, "b t"] | None = None  # left padded
    recur_pos_emb: at.Float[at.Array, "b t v p d2"] | None = None  # left padded
    recur_state_emb: at.Float[at.Array, "b t d3"] | None = None  # left padded

    # for Event-Graph 长时记忆（v1.3 §7）
    #
    # 这里传的是**原始**节点张量，不是已经聚合好的 512 个 token。聚合里有可训练的
    # 槽位读取（24 个 latent slot 对节点做软读取），只能在 forward 里算。
    # 长期段（n = max_nodes 个槽位）与近期段（r = recent_window 帧）分开传，
    # 由模型内部拼成 §7 的 384 + 128 = 512。
    #
    # 节点内部的 e 维、载荷的 p 维都是「采样帧 / 每帧 token」，两处 p 是同一个 16。
    # ⚠️ 轴名不要复用 d1：这里最后一维是**冻结 PCA token** 的 512，而下面 payload 的
    #    最后一维是**原始视觉 token** 的 2048。两个宽度不同，但 jaxtyping 的轴名是全局
    #    绑定的 —— 都叫 d1 会按先出现的那个（512）去卡 payload，直接类型报错。
    eg_node_tokens: at.Float[at.Array, "b n e p dt"] | None = None   # 冻结 PCA token，fp16
    eg_node_frame_mask: at.Bool[at.Array, "b n e"] | None = None
    eg_node_state: at.Float[at.Array, "b n e d3"] | None = None
    eg_node_mask: at.Bool[at.Array, "b n"] | None = None
    eg_node_mass: at.Float[at.Array, "b n"] | None = None             # 证据量（帧数）
    eg_node_bstr: at.Float[at.Array, "b n"] | None = None             # 边界强度
    eg_node_phase: at.Float[at.Array, "b n"] | None = None            # 0=video 1=exec
    eg_node_grasp: at.Float[at.Array, "b n"] | None = None            # 累计成功抓取数
    # 代表帧的**真实 2048-d** 载荷。§11 明确禁止用 512-d cache 顶替这一份。
    eg_node_payload: at.Float[at.Array, "b n p d1"] | None = None
    eg_node_payload_pos: at.Float[at.Array, "b n p d2"] | None = None
    eg_node_payload_state: at.Float[at.Array, "b n d3"] | None = None
    eg_recent_payload: at.Float[at.Array, "b r p d1"] | None = None
    eg_recent_payload_pos: at.Float[at.Array, "b r p d2"] | None = None
    eg_recent_payload_state: at.Float[at.Array, "b r d3"] | None = None
    eg_recent_mask: at.Bool[at.Array, "b r"] | None = None

    # for symbolic memory
    symbolic_tokenized_prompt: at.Int[at.Array, "b l2"] | None = None
    symbolic_tokenized_prompt_mask: at.Bool[at.Array, "b l2"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "HistAugObservation":
        parent_obs = super().from_dict(data)
        return cls(
            # Base observation fields
            images=parent_obs.images,
            image_masks=parent_obs.image_masks,
            state=parent_obs.state,
            tokenized_prompt=parent_obs.tokenized_prompt,
            tokenized_prompt_mask=parent_obs.tokenized_prompt_mask,
            token_ar_mask=parent_obs.token_ar_mask,
            token_loss_mask=parent_obs.token_loss_mask,
            progress=parent_obs.progress, # 加入的过程量
            # MMEVLA fields
            static_image_emb=data.get("static_image_emb", None),
            static_mask=data.get("static_mask", None),
            static_pos_emb=data.get("static_pos_emb", None),
            static_state_emb=data.get("static_state_emb", None),
            candidate_image_emb=data.get("candidate_image_emb", None),
            candidate_pos_emb=data.get("candidate_pos_emb", None),
            candidate_state_emb=data.get("candidate_state_emb", None),
            candidate_mask=data.get("candidate_mask", None),
            recur_image_emb=data.get("recur_image_emb", None),
            recur_mask=data.get("recur_mask", None),
            recur_pos_emb=data.get("recur_pos_emb", None),
            recur_state_emb=data.get("recur_state_emb", None),
            **{f: data.get(f, None) for f in EG_FIELDS},
            symbolic_tokenized_prompt=data.get("symbolic_tokenized_prompt", None),
            symbolic_tokenized_prompt_mask=data.get(
                "symbolic_tokenized_prompt_mask", None
            ),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        result = super().to_dict()
        result["static_image_emb"] = self.static_image_emb
        result["static_mask"] = self.static_mask
        result["static_pos_emb"] = self.static_pos_emb
        result["static_state_emb"] = self.static_state_emb
        result["candidate_image_emb"] = self.candidate_image_emb
        result["candidate_pos_emb"] = self.candidate_pos_emb
        result["candidate_state_emb"] = self.candidate_state_emb
        result["candidate_mask"] = self.candidate_mask
        result["recur_image_emb"] = self.recur_image_emb
        result["recur_mask"] = self.recur_mask
        result["recur_pos_emb"] = self.recur_pos_emb
        result["recur_state_emb"] = self.recur_state_emb
        for f in EG_FIELDS:
            result[f] = getattr(self, f)
        result["symbolic_tokenized_prompt"] = self.symbolic_tokenized_prompt  #  subgoal
        result["symbolic_tokenized_prompt_mask"] = self.symbolic_tokenized_prompt_mask
        return result

    def to_base_obs(self) -> _Observation:
        return _Observation(
            images=self.images,
            image_masks=self.image_masks,
            state=self.state,
            tokenized_prompt=self.tokenized_prompt,
            tokenized_prompt_mask=self.tokenized_prompt_mask,
            token_ar_mask=self.token_ar_mask,
            token_loss_mask=self.token_loss_mask,
            progress=self.progress, # 加入的过程量

        )

    @classmethod
    def from_base_obs(
        cls,
        base_obs: _Observation,
        static_image_emb: at.Float[ArrayT, "*b l d1"] | None = None,
        static_mask: at.Bool[ArrayT, "*b l"] | None = None,
        static_pos_emb: at.Float[ArrayT, "*b l d2"] | None = None,
        static_state_emb: at.Float[ArrayT, "*b l d3"] | None = None,
        candidate_image_emb: at.Float[ArrayT, "*b cw cp cd1"] | None = None,
        candidate_pos_emb: at.Float[ArrayT, "*b cw cp cd2"] | None = None,
        candidate_state_emb: at.Float[ArrayT, "*b cw cd3"] | None = None,
        candidate_mask: at.Bool[ArrayT, "*b cw"] | None = None,
        recur_image_emb: at.Float[ArrayT, "*b t v p d1"] | None = None,
        recur_mask: at.Bool[ArrayT, "*b t"] | None = None,
        recur_pos_emb: at.Float[ArrayT, "*b t v p d2"] | None = None,
        recur_state_emb: at.Float[ArrayT, "*b t d3"] | None = None,
        symbolic_tokenized_prompt: at.Int[ArrayT, "*b l d5"] | None = None,
        symbolic_tokenized_prompt_mask: at.Bool[ArrayT, "*b l d6"] | None = None,
        **eg,
    ) -> "HistAugObservation":
        """把 π0 的裸 Observation 升成 HistAugObservation。

        eg_* 那一批走 `**eg` 原样透传（它们的形状由 HistAugObservation 的字段注解
        把关），字段名写错会在构造时报 TypeError。名字的唯一定义见文件末尾 EG_FIELDS。
        """
        return HistAugObservation(
            **{f: getattr(base_obs, f) for f in _BASE_FIELDS},
            static_image_emb=static_image_emb,
            static_mask=static_mask,
            static_pos_emb=static_pos_emb,
            static_state_emb=static_state_emb,
            candidate_image_emb=candidate_image_emb,
            candidate_pos_emb=candidate_pos_emb,
            candidate_state_emb=candidate_state_emb,
            candidate_mask=candidate_mask,
            recur_image_emb=recur_image_emb,
            recur_mask=recur_mask,
            recur_pos_emb=recur_pos_emb,
            recur_state_emb=recur_state_emb,
            symbolic_tokenized_prompt=symbolic_tokenized_prompt,
            symbolic_tokenized_prompt_mask=symbolic_tokenized_prompt_mask,
            **eg,
        )


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: HistAugObservation,
    *args,
    **kwargs,
) -> HistAugObservation:
    base_obs: _Observation = _preprocess_observation(
        rng,
        observation.to_base_obs(),
        *args,
        **kwargs,
    )
    return HistAugObservation.from_base_obs(
        base_obs,
        static_image_emb=observation.static_image_emb,
        static_mask=observation.static_mask,
        static_pos_emb=observation.static_pos_emb,
        static_state_emb=observation.static_state_emb,
        candidate_image_emb=observation.candidate_image_emb,
        candidate_pos_emb=observation.candidate_pos_emb,
        candidate_state_emb=observation.candidate_state_emb,
        candidate_mask=observation.candidate_mask,
        recur_image_emb=observation.recur_image_emb,
        recur_mask=observation.recur_mask,
        recur_pos_emb=observation.recur_pos_emb,
        recur_state_emb=observation.recur_state_emb,
        symbolic_tokenized_prompt=observation.symbolic_tokenized_prompt,
        symbolic_tokenized_prompt_mask=observation.symbolic_tokenized_prompt_mask,
        # Event-Graph（v1.3）原样透传：节点 token 是冻结 PCA 出来的、代表帧载荷是
        # 冻结编码器出来的，都已经是特征，不能再过一次图像增广/归一化。
        **{f: getattr(observation, f) for f in EG_FIELDS},
    )


# Event-Graph 那批字段的**唯一名单**。它们要穿过四道白名单才能到模型：
#   ① RepackTransform（training/config.py）  ② RealRobotInputs（policies/realrobot_policy.py）
#   ③ HistAugObservation.from_dict/to_dict   ④ preprocess_observation
# 前两道和后两道都从这里取名单。少写一处的表现一律是「图是空的」——不报错，只是没记忆。
EG_FIELDS = (
    "eg_node_tokens",            # 冻结 PCA token        b n e p dt   (dt = event_dim = 512)
    "eg_node_frame_mask",        # 哪些采样帧是真的       b n e
    "eg_node_state",             # 逐帧归一化 state       b n e d3
    "eg_node_mask",              # 哪些槽位是真实节点     b n
    "eg_node_mass",              # 证据量（帧数）         b n
    "eg_node_bstr",              # 边界强度               b n
    "eg_node_phase",             # 0=video 1=exec         b n
    "eg_node_grasp",             # 累计成功抓取数         b n
    "eg_node_payload",           # 代表帧的原始 token     b n p d1   (d1 = 2048)
    "eg_node_payload_pos",       # 代表帧位置编码         b n p d2
    "eg_node_payload_state",     # 代表帧 state           b n d3
    "eg_recent_payload",         # 最近 r 帧原始 token    b r p d1
    "eg_recent_payload_pos",     # 最近 r 帧位置编码      b r p d2
    "eg_recent_payload_state",   # 最近 r 帧 state        b r d3
    "eg_recent_mask",            # 最近 r 帧里哪些有效    b r
)

# ⚠️ 轴名不要复用 d1：eg_node_tokens 的最后一维是**冻结 PCA token** 的 512，而 payload
#    那一支的最后一维是**原始视觉 token** 的 2048。两个宽度不同，但 jaxtyping 的轴名是
#    全局绑定的 —— 都叫 d1 会按先出现的那个（512）去卡 payload，直接类型报错。

# 类里 eg_ 开头的字段必须和上面这份名单一字不差。对不上就是「有人加了字段但忘了接
# 线」，那种情况下这四个环节会各挑自己认识的一部分传，图静默地少东西。这里让它在
# import 的时候就炸，而不是训到一半才发现图是空的。
assert set(EG_FIELDS) == {f for f in HistAugObservation.__dataclass_fields__ if f.startswith("eg_")}, \
    "EG_FIELDS 与 HistAugObservation 的 eg_* 字段不一致"

# 裸 π0 Observation 里要原样搬到 HistAugObservation 的那些字段。
# 不走 base_obs.to_dict()：它会顺手把 images/image_masks 改名成 image/image_mask。
_BASE_FIELDS = ("images", "image_masks", "state", "tokenized_prompt",
                "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask", "progress")
